"""player_tracker — слежение за одним футболистом на видео с iPhone.

Слои:
  * detection   — детектор людей (YOLO через ultralytics, лениво) и его заглушки;
  * multitracker— мультитрекер ByteTrack-типа (Калман + IoU + внешность);
  * appearance  — дескрипторы внешности (полосовые цветовые гистограммы или
                  нейросетевой ReID при наличии torchreid);
  * jersey      — распознавание номера на футболке и накопление голосов;
  * team        — разделение по цвету формы (своя команда / чужая / судья);
  * target      — модель целевого игрока и автомат состояний слежения с
                  повторным захватом после выхода из кадра или перекрытия;
  * video       — декодирование HEVC/HDR-роликов iPhone через ffmpeg;
  * pipeline    — склейка всего в один прогон; render/export — вывод.
"""

__all__ = [
    "appearance",
    "detection",
    "export",
    "geometry",
    "jersey",
    "kalman",
    "multitracker",
    "pipeline",
    "render",
    "target",
    "team",
    "video",
]
