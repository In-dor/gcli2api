"""
用于跟踪每个凭证文件API调用使用情况的统计模块。
使用更简单的逻辑：将当前时间与 next_reset_time 进行比较。
"""

import os
import time
from datetime import datetime, timezone, timedelta
from threading import Lock
from typing import Dict, Any, Optional

from config import get_credentials_dir, is_mongodb_mode
from log import log
from .state_manager import get_state_manager
from .storage_adapter import get_storage_adapter


def _get_next_utc_7am() -> datetime:
    """
    计算下一个 UTC 07:00 的配额重置时间。
    """
    now = datetime.now(timezone.utc)
    today_7am = now.replace(hour=7, minute=0, second=0, microsecond=0)

    if now < today_7am:
        return today_7am
    else:
        return today_7am + timedelta(days=1)


class UsageStats:
    """
    简化的使用情况统计管理器，具有清晰的重置逻辑。
    """

    def __init__(self):
        self._lock = Lock()
        # 状态文件路径将在初始化时异步设置
        self._state_file = None
        self._state_manager = None
        self._storage_adapter = None
        self._stats_cache: Dict[str, Dict[str, Any]] = {}
        self._initialized = False
        self._cache_dirty = False  # 缓存脏标记，减少不必要的写入
        self._last_save_time = 0
        self._save_interval = 60  # 最多每分钟保存一次，减少I/O
        self._max_cache_size = 100  # 严格限制缓存大小

    async def initialize(self):
        """初始化使用情况统计模块。"""
        if self._initialized:
            return

        # 初始化存储适配器
        self._storage_adapter = await get_storage_adapter()

        # 只在文件模式下创建本地状态文件
        if not await is_mongodb_mode():
            credentials_dir = await get_credentials_dir()
            self._state_file = os.path.join(credentials_dir, "creds_state.toml")
            self._state_manager = get_state_manager(self._state_file)

        await self._load_stats()
        self._initialized = True
        storage_type = "MongoDB" if await is_mongodb_mode() else "File"
        log.debug(f"使用情况统计模块已使用 {storage_type} 存储后端初始化")

    def _normalize_filename(self, filename: str) -> str:
        """将文件名规范化为相对路径，以实现一致的存储。"""
        if not filename:
            return ""

        if os.path.sep not in filename and "/" not in filename:
            return filename

        return os.path.basename(filename)

    def _is_gemini_2_5_pro(self, model_name: str) -> bool:
        """
        检查模型是否为 gemini-2.5-pro 的变体（包括前缀和后缀）。
        """
        if not model_name:
            return False

        try:
            from config import get_base_model_name, get_base_model_from_feature_model

            # Remove feature prefixes (流式抗截断/, 假流式/)
            base_with_suffix = get_base_model_from_feature_model(model_name)

            # 移除 thinking/search 后缀 (-maxthinking, -nothinking, -search)
            pure_base_model = get_base_model_name(base_with_suffix)

            # 检查纯基础模型是否正好是 "gemini-2.5-pro"
            return pure_base_model == "gemini-2.5-pro"

        except ImportError:
            # 如果配置导入失败，则使用回退逻辑
            clean_model = model_name
            for prefix in ["流式抗截断/", "假流式/"]:
                if clean_model.startswith(prefix):
                    clean_model = clean_model[len(prefix) :]
                    break

            for suffix in ["-maxthinking", "-nothinking", "-search"]:
                if clean_model.endswith(suffix):
                    clean_model = clean_model[: -len(suffix)]
                    break

            return clean_model == "gemini-2.5-pro"

    def _is_gemini_3_pro(self, model_name: str) -> bool:
        """
        检查模型是否为 gemini-3-pro 的变体。
        """
        if not model_name:
            return False

        try:
            from config import get_base_model_name, get_base_model_from_feature_model

            base_with_suffix = get_base_model_from_feature_model(model_name)
            pure_base_model = get_base_model_name(base_with_suffix)

            return "gemini-3-pro" in pure_base_model

        except ImportError:
            clean_model = model_name
            for prefix in ["流式抗截断/", "假流式/"]:
                if clean_model.startswith(prefix):
                    clean_model = clean_model[len(prefix) :]
                    break

            for suffix in ["-maxthinking", "-nothinking", "-search"]:
                if clean_model.endswith(suffix):
                    clean_model = clean_model[: -len(suffix)]
                    break

            return "gemini-3-pro" in clean_model

    async def _load_stats(self):
        """从统一存储加载统计信息"""
        try:
            # 从统一存储获取所有使用统计，添加超时机制防止卡死
            import asyncio

            async def load_stats_with_timeout():
                all_usage_stats = await self._storage_adapter.get_all_usage_stats()

                log.debug(f"正在处理 {len(all_usage_stats)} 个使用情况统计项...")

                # 直接处理统计数据
                stats_cache = {}
                processed_count = 0

                for filename, stats_data in all_usage_stats.items():
                    if isinstance(stats_data, dict):
                        normalized_filename = self._normalize_filename(filename)

                        # 提取使用统计字段
                        usage_data = {
                            "gemini_2_5_pro_calls": stats_data.get("gemini_2_5_pro_calls", 0),
                            "gemini_3_pro_calls": stats_data.get("gemini_3_pro_calls", 0),
                            "total_calls": stats_data.get("total_calls", 0),
                            "next_reset_time": stats_data.get("next_reset_time"),
                            "daily_limit_gemini_2_5_pro": stats_data.get(
                                "daily_limit_gemini_2_5_pro", 200
                            ),
                            "daily_limit_gemini_3_pro": stats_data.get(
                                "daily_limit_gemini_3_pro", 200
                            ),
                            "daily_limit_total": stats_data.get("daily_limit_total", 1000),
                        }

                        # 只加载有实际使用数据的统计，或者有reset时间的
                        if (
                            usage_data.get("gemini_2_5_pro_calls", 0) > 0
                            or usage_data.get("gemini_3_pro_calls", 0) > 0
                            or usage_data.get("total_calls", 0) > 0
                            or usage_data.get("next_reset_time")
                        ):
                            stats_cache[normalized_filename] = usage_data
                            processed_count += 1

                return stats_cache, processed_count

            # 设置15秒超时防止卡死
            try:
                self._stats_cache, processed_count = await asyncio.wait_for(
                    load_stats_with_timeout(), timeout=15.0
                )
                log.debug(f"已为 {processed_count} 个凭证文件加载使用情况统计")
            except asyncio.TimeoutError:
                log.error("加载使用情况统计在30秒后超时，使用空缓存")
                self._stats_cache = {}
                return

        except Exception as e:
            log.error(f"加载使用情况统计失败: {e}")
            self._stats_cache = {}

    async def _save_stats(self):
        """将统计信息保存到统一存储。"""
        current_time = time.time()

        # 使用脏标记和时间间隔控制，减少不必要的写入
        if not self._cache_dirty or (current_time - self._last_save_time < self._save_interval):
            return

        try:
            # 批量更新使用统计到存储适配器
            log.debug(f"正在保存 {len(self._stats_cache)} 个使用情况统计项...")

            saved_count = 0
            for filename, stats in self._stats_cache.items():
                try:
                    stats_data = {
                        "gemini_2_5_pro_calls": stats.get("gemini_2_5_pro_calls", 0),
                        "gemini_3_pro_calls": stats.get("gemini_3_pro_calls", 0),
                        "total_calls": stats.get("total_calls", 0),
                        "next_reset_time": stats.get("next_reset_time"),
                        "daily_limit_gemini_2_5_pro": stats.get("daily_limit_gemini_2_5_pro", 200),
                        "daily_limit_gemini_3_pro": stats.get("daily_limit_gemini_3_pro", 200),
                        "daily_limit_total": stats.get("daily_limit_total", 1000),
                    }

                    success = await self._storage_adapter.update_usage_stats(filename, stats_data)
                    if success:
                        saved_count += 1
                except Exception as e:
                    log.error(f"保存 {filename} 的统计信息失败: {e}")
                    continue

            self._cache_dirty = False  # 清除脏标记
            self._last_save_time = current_time
            log.debug(
                f"已成功将 {saved_count}/{len(self._stats_cache)} 个使用情况统计保存到统一存储"
            )
        except Exception as e:
            log.error(f"保存使用情况统计失败: {e}")

    def _get_or_create_stats(self, filename: str) -> Dict[str, Any]:
        """获取或创建凭证文件的统计条目。"""
        normalized_filename = self._normalize_filename(filename)

        if normalized_filename not in self._stats_cache:
            # 严格控制缓存大小 - 超过限制时删除最旧的条目
            if len(self._stats_cache) >= self._max_cache_size:
                # 删除最旧的统计数据（基于next_reset_time或没有该字段的）
                oldest_key = min(
                    self._stats_cache.keys(),
                    key=lambda k: self._stats_cache[k].get("next_reset_time", ""),
                )
                del self._stats_cache[oldest_key]
                self._cache_dirty = True
                log.debug(f"已移除最旧的使用情况统计缓存条目: {oldest_key}")

            next_reset = _get_next_utc_7am()
            self._stats_cache[normalized_filename] = {
                "gemini_2_5_pro_calls": 0,
                "gemini_3_pro_calls": 0,
                "total_calls": 0,
                "next_reset_time": next_reset.isoformat(),
                "daily_limit_gemini_2_5_pro": 100,
                "daily_limit_gemini_3_pro": 100,
                "daily_limit_total": 1000,
            }
            self._cache_dirty = True  # 标记缓存已修改

        return self._stats_cache[normalized_filename]

    def _check_and_reset_daily_quota(self, stats: Dict[str, Any]) -> bool:
        """
        简单的重置逻辑：如果当前时间 >= next_reset_time，则重置。
        """
        try:
            next_reset_str = stats.get("next_reset_time")
            if not next_reset_str:
                # 未记录下一次重置时间，进行设置
                next_reset = _get_next_utc_7am()
                stats["next_reset_time"] = next_reset.isoformat()
                return False

            next_reset = datetime.fromisoformat(next_reset_str)
            now = datetime.now(timezone.utc)

            # 简单比较：如果当前时间 >= 下一次重置时间，则重置
            if now >= next_reset:
                old_gemini_calls = stats.get("gemini_2_5_pro_calls", 0)
                old_gemini_3_calls = stats.get("gemini_3_pro_calls", 0)
                old_total_calls = stats.get("total_calls", 0)

                # 重置计数器并设置新的下一次重置时间
                new_next_reset = _get_next_utc_7am()
                stats.update(
                    {
                        "gemini_2_5_pro_calls": 0,
                        "gemini_3_pro_calls": 0,
                        "total_calls": 0,
                        "next_reset_time": new_next_reset.isoformat(),
                    }
                )

                self._cache_dirty = True  # 标记缓存已修改
                log.info(
                    f"已执行每日配额重置。先前统计 - 2.5 Pro: {old_gemini_calls}, 3.0 Pro: {old_gemini_3_calls}, 总计: {old_total_calls}"
                )
                return True

            return False
        except Exception as e:
            log.error(f"每日配额重置检查出错: {e}")
            return False

    async def record_successful_call(self, filename: str, model_name: str):
        """记录一次成功的 API 调用以进行统计。"""
        if not self._initialized:
            await self.initialize()

        with self._lock:
            try:
                normalized_filename = self._normalize_filename(filename)
                stats = self._get_or_create_stats(normalized_filename)

                # 检查并根据需要执行每日重置
                reset_performed = self._check_and_reset_daily_quota(stats)

                # 增加计数器
                is_gemini_2_5_pro = self._is_gemini_2_5_pro(model_name)
                is_gemini_3_pro = self._is_gemini_3_pro(model_name)

                stats["total_calls"] += 1
                if is_gemini_2_5_pro:
                    stats["gemini_2_5_pro_calls"] += 1
                elif is_gemini_3_pro:
                    # 初始化字段，如果不存在（向后兼容）
                    if "gemini_3_pro_calls" not in stats:
                        stats["gemini_3_pro_calls"] = 0
                    stats["gemini_3_pro_calls"] += 1

                self._cache_dirty = True  # 标记缓存已修改

                log.debug(
                    f"已记录使用情况 - 文件: {normalized_filename}, 模型: {model_name}, "
                    f"2.5 Pro: {stats['gemini_2_5_pro_calls']}/{stats.get('daily_limit_gemini_2_5_pro', 100)}, "
                    f"3.0 Pro: {stats.get('gemini_3_pro_calls', 0)}/{stats.get('daily_limit_gemini_3_pro', 100)}, "
                    f"总计: {stats['total_calls']}/{stats.get('daily_limit_total', 1000)}"
                )

                if reset_performed:
                    log.info(f"已为 {normalized_filename} 重置每日配额")

            except Exception as e:
                log.error(f"记录使用情况统计失败: {e}")

        # 异步保存统计信息
        try:
            await self._save_stats()
        except Exception as e:
            log.error(f"记录后保存使用情况统计失败: {e}")

    async def get_usage_stats(self, filename: str = None) -> Dict[str, Any]:
        """获取使用情况统计。"""
        if not self._initialized:
            await self.initialize()

        with self._lock:
            if filename:
                normalized_filename = self._normalize_filename(filename)
                stats = self._get_or_create_stats(normalized_filename)
                # 返回统计信息前检查每日重置
                self._check_and_reset_daily_quota(stats)
                return {
                    "filename": normalized_filename,
                    "gemini_2_5_pro_calls": stats.get("gemini_2_5_pro_calls", 0),
                    "gemini_3_pro_calls": stats.get("gemini_3_pro_calls", 0),
                    "total_calls": stats.get("total_calls", 0),
                    "daily_limit_gemini_2_5_pro": stats.get("daily_limit_gemini_2_5_pro", 200),
                    "daily_limit_gemini_3_pro": stats.get("daily_limit_gemini_3_pro", 200),
                    "daily_limit_total": stats.get("daily_limit_total", 1000),
                    "next_reset_time": stats.get("next_reset_time"),
                }
            else:
                # 返回所有统计信息
                all_stats = {}
                for filename, stats in self._stats_cache.items():
                    # 检查每个文件的每日重置
                    self._check_and_reset_daily_quota(stats)
                    all_stats[filename] = {
                        "gemini_2_5_pro_calls": stats.get("gemini_2_5_pro_calls", 0),
                        "gemini_3_pro_calls": stats.get("gemini_3_pro_calls", 0),
                        "total_calls": stats.get("total_calls", 0),
                        "daily_limit_gemini_2_5_pro": stats.get("daily_limit_gemini_2_5_pro", 200),
                        "daily_limit_gemini_3_pro": stats.get("daily_limit_gemini_3_pro", 200),
                        "daily_limit_total": stats.get("daily_limit_total", 1000),
                        "next_reset_time": stats.get("next_reset_time"),
                    }

                return all_stats

    async def get_aggregated_stats(self) -> Dict[str, Any]:
        """获取所有凭证文件的聚合统计信息。"""
        if not self._initialized:
            await self.initialize()

        all_stats = await self.get_usage_stats()

        total_gemini_2_5_pro = 0
        total_gemini_3_pro = 0
        total_all_models = 0
        total_files = len(all_stats)

        for stats in all_stats.values():
            total_gemini_2_5_pro += stats.get("gemini_2_5_pro_calls", 0)
            total_gemini_3_pro += stats.get("gemini_3_pro_calls", 0)
            total_all_models += stats.get("total_calls", 0)

        return {
            "total_files": total_files,
            "total_gemini_2_5_pro_calls": total_gemini_2_5_pro,
            "total_gemini_3_pro_calls": total_gemini_3_pro,
            "total_all_model_calls": total_all_models,
            "avg_gemini_2_5_pro_per_file": total_gemini_2_5_pro / max(total_files, 1),
            "avg_gemini_3_pro_per_file": total_gemini_3_pro / max(total_files, 1),
            "avg_total_per_file": total_all_models / max(total_files, 1),
            "next_reset_time": _get_next_utc_7am().isoformat(),
        }

    async def update_daily_limits(
        self,
        filename: str,
        gemini_2_5_pro_limit: int = None,
        gemini_3_pro_limit: int = None,
        total_limit: int = None,
    ):
        """更新特定凭证文件的每日限制。"""
        if not self._initialized:
            await self.initialize()

        with self._lock:
            try:
                normalized_filename = self._normalize_filename(filename)
                stats = self._get_or_create_stats(normalized_filename)

                if gemini_2_5_pro_limit is not None:
                    stats["daily_limit_gemini_2_5_pro"] = gemini_2_5_pro_limit

                if gemini_3_pro_limit is not None:
                    stats["daily_limit_gemini_3_pro"] = gemini_3_pro_limit

                if total_limit is not None:
                    stats["daily_limit_total"] = total_limit

                log.info(
                    f"已更新 {normalized_filename} 的每日限制: "
                    f"2.5 Pro = {stats.get('daily_limit_gemini_2_5_pro', 100)}, "
                    f"3.0 Pro = {stats.get('daily_limit_gemini_3_pro', 100)}, "
                    f"总计 = {stats.get('daily_limit_total', 1000)}"
                )

            except Exception as e:
                log.error(f"更新每日限制失败: {e}")
                raise

        await self._save_stats()

    async def reset_stats(self, filename: str = None):
        """重置使用情况统计。"""
        if not self._initialized:
            await self.initialize()

        with self._lock:
            if filename:
                normalized_filename = self._normalize_filename(filename)
                if normalized_filename in self._stats_cache:
                    # 手动重置：重置计数器并设置新的下一次重置时间
                    next_reset = _get_next_utc_7am()
                    self._stats_cache[normalized_filename].update(
                        {
                            "gemini_2_5_pro_calls": 0,
                            "gemini_3_pro_calls": 0,
                            "total_calls": 0,
                            "next_reset_time": next_reset.isoformat(),
                        }
                    )
                    log.info(f"已重置 {normalized_filename} 的使用情况统计")
            else:
                # 重置所有统计信息
                next_reset = _get_next_utc_7am()
                for filename, stats in self._stats_cache.items():
                    stats.update(
                        {
                            "gemini_2_5_pro_calls": 0,
                            "gemini_3_pro_calls": 0,
                            "total_calls": 0,
                            "next_reset_time": next_reset.isoformat(),
                        }
                    )
                log.info("已重置所有凭证文件的使用情况统计")

        await self._save_stats()


# 全局实例
_usage_stats_instance: Optional[UsageStats] = None


async def get_usage_stats_instance() -> UsageStats:
    """获取全局使用情况统计实例。"""
    global _usage_stats_instance
    if _usage_stats_instance is None:
        _usage_stats_instance = UsageStats()
        await _usage_stats_instance.initialize()
    return _usage_stats_instance


async def record_successful_call(filename: str, model_name: str):
    """用于记录成功 API 调用的便捷函数。"""
    stats = await get_usage_stats_instance()
    await stats.record_successful_call(filename, model_name)


async def get_usage_stats(filename: str = None) -> Dict[str, Any]:
    """用于获取使用情况统计的便捷函数。"""
    stats = await get_usage_stats_instance()
    return await stats.get_usage_stats(filename)


async def get_aggregated_stats() -> Dict[str, Any]:
    """用于获取聚合统计信息的便捷函数。"""
    stats = await get_usage_stats_instance()
    return await stats.get_aggregated_stats()
