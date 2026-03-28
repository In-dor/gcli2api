"""
日志路由模块 - 处理 /logs/* 相关的HTTP请求和WebSocket连接
"""

import asyncio
import datetime
import os

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from starlette.websockets import WebSocketState

import config
from log import log
from src.log_realtime import log_realtime_channel
from src.utils import verify_panel_token
from .utils import ConnectionManager


# 创建路由器
router = APIRouter(prefix="/logs", tags=["logs"])

# 心跳消息（用于防止长时间无日志时连接被中间网络设备回收）
HEARTBEAT_MESSAGE = "__GCLI2API_LOG_HEARTBEAT__"
HEARTBEAT_INTERVAL = 10

# WebSocket连接管理器
manager = ConnectionManager(max_connections=20)


@router.post("/clear")
async def clear_logs(token: str = Depends(verify_panel_token)):
    """清空日志文件"""
    try:
        # 直接使用环境变量获取日志文件路径
        log_file_path = os.getenv("LOG_FILE", "log.txt")

        # 检查日志文件是否存在
        if os.path.exists(log_file_path):
            try:
                # 清空文件内容（保留文件），确保以UTF-8编码写入
                # 使用 with 确保文件正确关闭
                with open(log_file_path, "w", encoding="utf-8") as f:
                    f.write("")
                    f.flush()  # 强制刷新到磁盘
                    # with 退出时会自动关闭文件
                log.info(f"日志文件已清空: {log_file_path}")

                # 通知所有WebSocket连接日志已清空
                log_realtime_channel.publish("--- 日志文件已清空 ---")

                return JSONResponse(
                    content={"message": f"日志文件已清空: {os.path.basename(log_file_path)}"}
                )
            except Exception as e:
                log.error(f"清空日志文件失败: {e}")
                raise HTTPException(status_code=500, detail=f"清空日志文件失败: {str(e)}")
        else:
            return JSONResponse(content={"message": "日志文件不存在"})

    except Exception as e:
        log.error(f"清空日志文件失败: {e}")
        raise HTTPException(status_code=500, detail=f"清空日志文件失败: {str(e)}")


@router.get("/download")
async def download_logs(token: str = Depends(verify_panel_token)):
    """下载日志文件"""
    try:
        # 直接使用环境变量获取日志文件路径
        log_file_path = os.getenv("LOG_FILE", "log.txt")

        # 检查日志文件是否存在
        if not os.path.exists(log_file_path):
            raise HTTPException(status_code=404, detail="日志文件不存在")

        # 检查文件是否为空
        file_size = os.path.getsize(log_file_path)
        if file_size == 0:
            raise HTTPException(status_code=404, detail="日志文件为空")

        # 生成文件名（包含时间戳）
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"gcli2api_logs_{timestamp}.txt"

        log.info(f"下载日志文件: {log_file_path}")

        return FileResponse(
            path=log_file_path,
            filename=filename,
            media_type="text/plain",
            headers={"Content-Disposition": f"attachment; filename={filename}"},
        )

    except HTTPException:
        raise
    except Exception as e:
        log.error(f"下载日志文件失败: {e}")
        raise HTTPException(status_code=500, detail=f"下载日志文件失败: {str(e)}")


@router.websocket("/stream")
async def websocket_logs(websocket: WebSocket):
    """WebSocket端点，用于实时日志流"""
    # WebSocket 认证: 从查询参数获取 token
    token = websocket.query_params.get("token")

    if not token:
        await websocket.close(code=403, reason="Missing authentication token")
        log.warning("WebSocket连接被拒绝: 缺少认证token")
        return

    # 验证 token
    try:
        panel_password = await config.get_panel_password()
        if token != panel_password:
            await websocket.close(code=403, reason="Invalid authentication token")
            log.warning("WebSocket连接被拒绝: token验证失败")
            return
    except Exception as e:
        await websocket.close(code=1011, reason="Authentication error")
        log.error(f"WebSocket认证过程出错: {e}")
        return

    # 检查连接数限制
    if not await manager.connect(websocket):
        return

    try:
        # 直接使用环境变量获取日志文件路径
        log_file_path = os.getenv("LOG_FILE", "log.txt")

        # 发送初始日志（限制为最后50行，减少内存占用）
        if os.path.exists(log_file_path):
            try:
                # 使用 with 确保文件正确关闭
                with open(log_file_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    # 只发送最后50行，减少初始内存消耗
                    for line in lines[-50:]:
                        if line.strip():
                            await websocket.send_text(line.strip())
            except Exception as e:
                await websocket.send_text(f"Error reading log file: {e}")
                log.error(f"WebSocket初始日志读取错误: {e}")

        # 订阅日志实时发布通道
        subscriber_id, subscriber_queue = await log_realtime_channel.subscribe()

        # 创建后台任务监听客户端断开
        # 即使没有日志更新，receive_text() 也能即时感知断开
        async def listen_for_disconnect():
            try:
                while True:
                    await websocket.receive_text()
            except Exception:
                pass

        listener_task = asyncio.create_task(listen_for_disconnect())

        try:
            while websocket.client_state == WebSocketState.CONNECTED:
                queue_task = asyncio.create_task(subscriber_queue.get())
                done, pending = await asyncio.wait(
                    [listener_task, queue_task],
                    timeout=HEARTBEAT_INTERVAL,
                    return_when=asyncio.FIRST_COMPLETED,
                )

                # 如果监听任务结束（通常是因为连接断开），则退出循环
                if listener_task in done:
                    if not queue_task.done():
                        queue_task.cancel()
                        try:
                            await queue_task
                        except asyncio.CancelledError:
                            pass
                    break

                # 收到实时日志，立即推送
                if queue_task in done:
                    try:
                        message = queue_task.result()
                        await websocket.send_text(message)
                    except Exception:
                        break
                else:
                    # 定期发送心跳，避免长期空闲导致连接被中间网络设备回收
                    try:
                        await websocket.send_text(HEARTBEAT_MESSAGE)
                    except Exception:
                        break

                # queue_task 每轮都会新建，若未完成需及时取消以免泄漏
                if queue_task in pending:
                    queue_task.cancel()
                    try:
                        await queue_task
                    except asyncio.CancelledError:
                        pass

        finally:
            # 确保清理监听任务
            if not listener_task.done():
                listener_task.cancel()
                try:
                    await listener_task
                except asyncio.CancelledError:
                    pass
            log_realtime_channel.unsubscribe(subscriber_id)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        log.error(f"WebSocket logs error: {e}")
    finally:
        manager.disconnect(websocket)
