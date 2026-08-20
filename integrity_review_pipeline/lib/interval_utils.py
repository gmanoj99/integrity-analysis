"""Interval helpers for session-offset ranges (milliseconds)."""

from __future__ import annotations

MsInterval = tuple[int, int]


def union_interval_length_ms(intervals: list[MsInterval]) -> int:
    """Union length of half-open intervals [start, end). Overlaps merge."""
    if not intervals:
        return 0
    sorted_intervals = sorted((a, b) for a, b in intervals if b > a)
    if not sorted_intervals:
        return 0

    total = 0
    cur_start, cur_end = sorted_intervals[0]
    for a, b in sorted_intervals[1:]:
        if a <= cur_end:
            cur_end = max(cur_end, b)
        else:
            total += cur_end - cur_start
            cur_start, cur_end = a, b
    total += cur_end - cur_start
    return total


def _merge_intervals(intervals: list[MsInterval]) -> list[tuple[int, int]]:
    sorted_intervals = sorted((a, b) for a, b in intervals if b > a)
    if not sorted_intervals:
        return []
    out: list[tuple[int, int]] = []
    cur_start, cur_end = sorted_intervals[0]
    for a, b in sorted_intervals[1:]:
        if a <= cur_end:
            cur_end = max(cur_end, b)
        else:
            out.append((cur_start, cur_end))
            cur_start, cur_end = a, b
    out.append((cur_start, cur_end))
    return out


def intersect_union_length_ms(a: list[MsInterval], b: list[MsInterval]) -> int:
    """Length of intersection of two interval unions."""
    left = _merge_intervals(a)
    right = _merge_intervals(b)
    if not left or not right:
        return 0

    total = 0
    i = j = 0
    while i < len(left) and j < len(right):
        a0, a1 = left[i]
        b0, b1 = right[j]
        start = max(a0, b0)
        end = min(a1, b1)
        if end > start:
            total += end - start
        if a1 < b1:
            i += 1
        else:
            j += 1
    return total
