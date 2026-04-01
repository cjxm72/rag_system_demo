from __future__ import annotations

import threading
from functools import wraps
from typing import Any, Callable, Dict, Hashable, Optional, Tuple, TypeVar, cast

T = TypeVar("T")


def _default_key(args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Hashable:
    # 允许按“配置”区分不同单例（例如不同 model_id / base_url / api_key）。
    # kwargs 排序后纳入 key，保证稳定。
    return (args, tuple(sorted(kwargs.items(), key=lambda x: x[0])))


def singleton(
    func: Optional[Callable[..., T]] = None,
    *,
    key: Optional[Callable[..., Hashable]] = None,
) -> Callable[..., T]:
    """
    单例装饰器：
    - 默认按 (args, kwargs) 形成 key，从而“每套配置一个全局唯一实例”
    - 可通过 key= 自定义缓存 key
    """

    def decorator(f: Callable[..., T]) -> Callable[..., T]:
        lock = threading.Lock()
        cache: Dict[Hashable, T] = {}

        @wraps(f)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            k = key(*args, **kwargs) if key is not None else _default_key(args, kwargs)
            if k in cache:
                return cache[k]
            with lock:
                if k in cache:
                    return cache[k]
                inst = f(*args, **kwargs)
                cache[k] = inst
                return inst

        return cast(Callable[..., T], wrapper)

    if func is None:
        return decorator
    return decorator(func)

