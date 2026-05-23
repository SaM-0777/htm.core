from datetime import datetime


def parse_time(
    recorded_time: str,
) -> dict:

    try:

        dt = datetime.fromisoformat(recorded_time)

    except Exception:

        #
        # Never crash parser pipeline
        #

        return {
            "year": None,
            "month": None,
            "day": None,
            "hour": None,
            "minute": None,
        }

    return {
        "year": dt.year,
        "month": dt.month,
        "day": dt.day,
        "hour": dt.hour,
        "minute": dt.minute,
    }
