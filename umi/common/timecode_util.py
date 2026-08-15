from typing import Union
from fractions import Fraction
import datetime
import av


def timecode_to_seconds(
        timecode: str, frame_rate: Union[int, float, Fraction]
        ) -> Union[float, Fraction]:
    """
    Convert timecode into seconds since midnight.

    A semicolon before the frame number indicates drop-frame timecode,
    e.g. GoPro 59.94fps videos can report HH:MM:SS;FF.
    """
    # calculate whole frame rate
    # 29.97 -> 30, 59.94 -> 60
    int_frame_rate = round(frame_rate)

    # parse timecode string
    is_drop_frame = ';' in timecode
    h, m, s, f = [int(x) for x in timecode.replace(';', ':').split(':')]

    # calculate frames assuming whole frame rate
    frames = (3600 * h + 60 * m + s) * int_frame_rate + f
    if is_drop_frame:
        # SMPTE drop-frame timecode skips frame numbers at the start of
        # each minute, except every tenth minute.
        drop_frames = round(int_frame_rate * 0.066666)
        total_minutes = 60 * h + m
        frames -= drop_frames * (total_minutes - total_minutes // 10)

    # convert to seconds
    seconds = frames / frame_rate
    return seconds


def stream_get_start_datetime(stream: av.stream.Stream) -> datetime.datetime:
    """
    Combines creation time and timecode to get high-precision
    time for the first frame of a video.
    """
    # read metadata
    frame_rate = stream.average_rate
    tc = stream.metadata['timecode']
    creation_time = stream.metadata['creation_time']
    
    # get time within the day
    seconds_since_midnight = float(timecode_to_seconds(timecode=tc, frame_rate=frame_rate))
    delta = datetime.timedelta(seconds=seconds_since_midnight)
    
    # get dates
    create_datetime = datetime.datetime.strptime(creation_time, r"%Y-%m-%dT%H:%M:%S.%fZ")
    create_datetime = create_datetime.replace(hour=0, minute=0, second=0, microsecond=0)
    start_datetime = create_datetime + delta
    return start_datetime


def mp4_get_start_datetime(mp4_path: str) -> datetime.datetime:
    with av.open(mp4_path) as container:
        stream = container.streams.video[0]
        return stream_get_start_datetime(stream=stream)
