import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from PySide6.QtCore import QSize, Qt
from PySide6.QtCore import QObject, QThread, QTimer, Signal, Slot, QUrl
from PySide6.QtGui import QCursor, QWheelEvent
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QStyle,
    QVBoxLayout,
    QWidget,
)

from __feature__ import snake_case, true_property

class SeparationWorker(QObject):
    finished = Signal(dict)
    failed = Signal(str)

    def __init__(self, source_path: Path, output_dir: Path) -> None:
        super().__init__()
        self.source_path = source_path
        self.output_dir = output_dir

    @Slot()
    def run(self) -> None:
        try:
            from demucs.api import Separator
            from demucs.audio import save_audio

            # The six-source model includes guitar and piano in addition to
            # vocals, drums, bass, and other.
            separator = Separator(model="htdemucs_6s")
            _, separated = separator.separate_audio_file(str(self.source_path))
            model_dir = self.output_dir / self.source_path.stem
            model_dir.mkdir(parents=True, exist_ok=True)

            stem_paths: dict[str, str] = {}
            # Demucs returns a mapping from source names to tensors.  Source
            # names belong to the loaded model; Separator itself has no
            # ``sources`` attribute in the current API.
            for source, audio in separated.items():
                destination = model_dir / f"{source}.wav"
                save_audio(audio, str(destination), samplerate=separator.samplerate)
                stem_paths[source] = str(destination)
            self.finished.emit(stem_paths)
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class VolumeSlider(QSlider):
    def wheelEvent(self, event: QWheelEvent) -> None:
        # Let the enclosing scroll area handle wheel and trackpad scrolling.
        event.ignore()


class StemRow(QFrame):
    def __init__(self, stem_name: str, accent: str) -> None:
        super().__init__()
        self.object_name = "stemRow"
        self.set_fixed_height(76)

        layout = QHBoxLayout(self)
        layout.content_margins = 18, 12, 18, 12
        layout.spacing = 16

        name = QLabel(stem_name)
        name.object_name = "stemName"
        name.style_sheet = f"color: {accent};"
        name.set_fixed_width(110)
        name.minimum_height = (26)

        self.slider = VolumeSlider(Qt.Orientation.Horizontal)
        self.slider.cursor = QCursor(Qt.CursorShape.PointingHandCursor)
        self.slider.range = 0, 100
        self.slider.value = 100
        self.slider.accessible_name = f"{stem_name} volume"
        self.slider.minimum_size = QSize(180, 32)

        self.volume_label = QLabel("100%")
        self.volume_label.object_name = "volumeLabel"
        self.volume_label.set_fixed_width(56)
        self.volume_label.minimum_height = 26
        self.slider.valueChanged.connect(
            lambda value: setattr(self.volume_label, "text", f"{value}%")
        )

        self.mute_button = QPushButton("MUTE")
        self.mute_button.checkable = True
        self.mute_button.minimum_size = QSize(100, 38)
        self.mute_button.cursor = QCursor(Qt.CursorShape.PointingHandCursor)
        self.mute_button.accessible_name = f"Mute {stem_name}"
        self.mute_button.toggled.connect(
            lambda muted: setattr(self.mute_button, "text", "UNMUTE" if muted else "MUTE")
        )

        layout.add_widget(name)
        layout.add_widget(self.slider, 1)
        layout.add_widget(self.volume_label)
        layout.add_widget(self.mute_button)


class MixerWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.title = "Stem Mixer"
        self.minimum_size = QSize(800, 650)
        self.resize(960, 900)
        self.track_path: Path | None = None
        self.stem_paths: dict[str, str] = {}
        self.output_dir = Path(tempfile.mkdtemp(prefix="stem-mixer-"))
        self.worker_thread: QThread | None = None
        self.worker: SeparationWorker | None = None
        self.stem_rows: dict[str, StemRow] = {}
        self.mix_path: Path | None = None
        self.mix_timer = QTimer(self)
        self.mix_timer.single_shot = True
        self.mix_timer.interval = 120
        self.mix_timer.timeout.connect(self.render_mix)

        self.playback_players: dict[str, QMediaPlayer] = {}
        self.playback_outputs: dict[str, QAudioOutput] = {}
        self.playback_master: QMediaPlayer | None = None

        central_widget = QWidget()
        central_widget.object_name = "app"
        self.central_widget = central_widget
        layout = QVBoxLayout(central_widget)
        layout.content_margins = (36, 30, 36, 24)
        layout.spacing = 22

        header = QHBoxLayout()
        title_block = QVBoxLayout()
        title = QLabel("STEM MIXER")
        title.object_name = "title"
        subtitle = QLabel("Shape the balance of every layer.")
        subtitle.object_name = "subtitle"
        title_block.add_widget(title)
        title_block.add_widget(subtitle)

        brand = QLabel("MIX / 01")
        brand.object_name = "brand"
        header.add_layout(title_block)
        header.add_stretch()
        self.back_button = QPushButton("Back to upload")
        self.back_button.object_name = "secondaryButton"
        self.back_button.cursor = QCursor(Qt.CursorShape.PointingHandCursor)
        self.back_button.clicked.connect(self.reset_to_upload)
        self.back_button.hide()
        header.add_widget(self.back_button)
        header.add_widget(brand, alignment=Qt.AlignmentFlag.AlignTop)
        layout.add_layout(header)

        source_card = QFrame()
        source_card.object_name = "card"
        source_layout = QVBoxLayout(source_card)
        source_layout.content_margins = (22, 18, 22, 18)
        source_layout.spacing = 14

        source_heading = QLabel("SOURCE TRACK")
        source_heading.objecct_name = "sectionHeading"
        source_layout.add_widget(source_heading)

        source_row = QHBoxLayout()
        self.track_label = QLabel("No track selected")
        self.track_label.object_name = "trackLabel"
        self.track_label.minimum_width = 0
        self.track_label.text_interaction_flags = Qt.TextInteractionFlag.TextSelectableByMouse
        

        self.upload_button = QPushButton("Choose audio file")
        self.upload_button.object_name = "primaryButton"
        self.upload_button.cursor = QCursor(Qt.CursorShape.PointingHandCursor)
        self.upload_button.clicked.connect(self.choose_track)
        source_row.add_widget(self.track_label, 1)
        source_row.add_widget(self.upload_button)
        source_layout.add_layout(source_row)

        self.track_detail = QLabel(
            "Supported formats: WAV, MP3, FLAC, M4A, OGG"
        )
        self.track_detail.object_name = "mutedText"
        source_layout.add_widget(self.track_detail)
        self.separation_progress = QProgressBar()
        self.separation_progress.object_name = "separationProgress"
        self.separation_progress.range = (0, 0)
        self.separation_progress.is_text_visible = False
        self.separation_progress.fixed_height = 7
        self.separation_progress.hide()
        source_layout.add_widget(self.separation_progress)
        layout.add_widget(source_card)

        self.stems_header = QHBoxLayout()
        self.stems_title = QLabel("STEM CONTROLS")
        self.stems_title.object_name = "sectionHeading"
        self.stems_header.add_widget(self.stems_title)
        self.stems_header.add_stretch()
        self.separate_button = QPushButton("Separate stems")
        self.separate_button.object_name = "secondaryButton"
        self.separate_button.cursor = QCursor(Qt.CursorShape.PointingHandCursor)
        self.separate_button.clicked.connect(self.separate_stems)
        self.stems_header.add_widget(self.separate_button)
        layout.add_layout(self.stems_header)

        stems_card = QFrame()
        stems_card.object_name = "card"
        stems_layout = QVBoxLayout(stems_card)
        stems_layout.content_margins = (14, 8, 14, 8)
        stems_layout.spacing = 8

        for stem_name, key, accent in (
            ("Vocals", "vocals", "#f0a36b"),
            ("Drums", "drums", "#e7c56e"),
            ("Bass", "bass", "#75c7a1"),
            ("Guitar", "guitar", "#d28bea"),
            ("Piano", "piano", "#e58aa8"),
            ("Other", "other", "#8bacee"),
        ):
            row = StemRow(stem_name, accent)
            row.slider.valueChanged.connect(self.schedule_mix)
            row.mute_button.toggled.connect(self.schedule_mix)
            row.slider.valueChanged.connect(self.update_playback_volumes)
            row.mute_button.toggled.connect(self.update_playback_volumes)
            self.stem_rows[key] = row
            stems_layout.add_widget(row)
        self.stems_scroll = QScrollArea()
        self.stems_scroll.object_name = "stemsScroll"
        self.stems_scroll.widget_resizable = True
        self.stems_scroll.frame_shape = QFrame.Shape.NoFrame
        self.stems_scroll.minimum_height = 200
        self.stems_scroll.set_widget(stems_card)
        layout.add_widget(self.stems_scroll, 1)

        self.notice = QLabel("Choose a track, separate its stems, then adjust the mix.")
        self.notice.object_name = "notice"
        self.notice.word_wrap = True
        layout.add_widget(self.notice)

        self.footer = QLabel("Upload a track to get started")
        self.footer.object_name = "footer"
        layout.add_widget(self.footer)

        self.playback_seek_row = QHBoxLayout()
        self.elapsed_label = QLabel("0:00")
        self.elapsed_label.object_name = "mutedText"
        self.elapsed_label.fixed_width = 42

        self.position_slider = QSlider(Qt.Orientation.Horizontal)
        self.position_slider.object_name = "scrubber"
        self.position_slider.set_range(0, 0)
        self.position_slider.cursor = QCursor(Qt.CursorShape.PointingHandCursor)
        self.position_slider.accessible_name = "Mix playback position"
        self.position_slider.sliderMoved.connect(self.seek_mix)

        self.duration_label = QLabel("0:00")
        self.duration_label.object_name = "mutedText"
        self.duration_label.fixed_width = 42

        self.playback_seek_row.add_widget(self.elapsed_label)
        self.playback_seek_row.add_widget(self.position_slider, 1)
        self.playback_seek_row.add_widget(self.duration_label)
        layout.add_layout(self.playback_seek_row)

        self.playback_row = QHBoxLayout()
        self.playback_row.alignment = Qt.AlignmentFlag.AlignCenter

        self.play_button = QPushButton()
        self.play_button.object_name = "secondaryButton"
        self.play_button.icon = self.style().standard_icon(
            QStyle.StandardPixmap.SP_MediaPlay
        )
        self.play_button.icon_size = QSize(20, 20)
        self.play_button.fixed_size = QSize(46, 40)
        self.play_button.accessible_name = "Play mix"
        self.play_button.tool_tip = "Play mix"
        self.play_button.cursor = QCursor(Qt.CursorShape.PointingHandCursor)
        self.play_button.clicked.connect(self.play_mix)

        self.pause_button = QPushButton()
        self.pause_button.object_name = "secondaryButton"
        self.pause_button.icon = self.style().standard_icon(
            QStyle.StandardPixmap.SP_MediaPause
        )
        self.pause_button.icon_size = QSize(20, 20)
        self.pause_button.fixed_size = QSize(46, 40)
        self.pause_button.accessible_name = "Pause mix"
        self.pause_button.tool_tip = "Pause mix"
        self.pause_button.cursor = QCursor(Qt.CursorShape.PointingHandCursor)
        self.pause_button.clicked.connect(self.pause_mix)

        self.stop_button = QPushButton()
        self.stop_button.object_name = "secondaryButton"
        self.stop_button.icon = self.style().standard_icon(
            QStyle.StandardPixmap.SP_MediaStop
        )
        self.stop_button.icon_size = QSize(20, 20)
        self.stop_button.fixed_size = QSize(46, 40)
        self.stop_button.accessible_name = "Stop mix"
        self.stop_button.tool_tip = "Stop mix"
        self.stop_button.cursor = QCursor(Qt.CursorShape.PointingHandCursor)
        self.stop_button.clicked.connect(self.stop_mix)

        self.set_playback_controls_enabled(False)

        self.playback_row.add_widget(self.play_button)
        self.playback_row.add_widget(self.pause_button)
        self.playback_row.add_widget(self.stop_button)
        layout.add_layout(self.playback_row)

        self.set_mixer_content_visible(False)
        self.set_central_widget(central_widget)

        self.style_sheet = """
            QWidget#app {
                background: #111419;
                color: #eef0f4;
                font-family: "Avenir Next", "Helvetica Neue", sans-serif;
            }
            QLabel#title {
                color: #f5f6f8;
                font-size: 26px;
                font-weight: 800;
                letter-spacing: 2px;
            }
            QLabel#subtitle, QLabel#mutedText, QLabel#footer {
                color: #8d929c;
                font-size: 13px;
            }
            QLabel#brand {
                color: #72d6bd;
                font-size: 12px;
                font-weight: 700;
                letter-spacing: 1px;
            }
            QFrame#card {
                background: #191d24;
                border: 1px solid #292e37;
                border-radius: 12px;
            }
            QLabel#sectionHeading {
                color: #9da3ad;
                font-size: 11px;
                font-weight: 700;
                letter-spacing: 1.4px;
            }
            QLabel#trackLabel {
                color: #e8ebef;
                font-size: 14px;
            }
            QFrame#stemRow {
                background: #20252d;
                border-radius: 8px;
            }
            QScrollArea#stemsScroll {
                background: #191d24;
            }
            QProgressBar#separationProgress {
                background: #282e37;
                border: none;
                border-radius: 3px;
            }
            QProgressBar#separationProgress::chunk {
                background: #72d6bd;
                border-radius: 3px;
            }
            QLabel#stemName {
                font-size: 16px;
                font-weight: 700;
            }
            QLabel#volumeLabel {
                color: #eef0f4;
                font-size: 14px;
            }
            QSlider::groove:horizontal {
                height: 5px;
                background: #3a404a;
                border-radius: 2px;
            }
            QSlider::sub-page:horizontal {
                background: #72d6bd;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                width: 14px;
                margin: -5px 0;
                background: #f4f6f7;
                border: 2px solid #72d6bd;
                border-radius: 7px;
            }
            QPushButton {
                background: #282e37;
                color: #eef0f4;
                border: 1px solid #383f4a;
                padding: 9px 14px;
                border-radius: 7px;
                font-size: 12px;
                font-weight: 700;
            }
            QPushButton:hover {
                background: #333a45;
            }
            QPushButton#primaryButton {
                background: #72d6bd;
                color: #101714;
                border: none;
            }
            QPushButton#primaryButton:hover {
                background: #8be2cc;
            }
            QPushButton#secondaryButton {
                background: #282e37;
                color: #d7dbe2;
                border: 1px solid #383f4a;
            }
            QPushButton#secondaryButton:hover {
                background: #333a45;
            }
            QPushButton:checked {
                background: #583b40;
                color: #ffb7b0;
                border: 1px solid #80545a;
            }
            QLabel#notice {
                color: #a7adba;
                background: #191d24;
                border-left: 3px solid #72d6bd;
                padding: 11px 14px;
                font-size: 12px;
            }
            """

    
    def choose_track(self) -> None:
        file_path, _ = QFileDialog.get_open_file_name(
            self,
            "Choose an audio track",
            "",
            "Audio files (*.wav *.mp3 *.flac *.m4a *.ogg);;All files (*)",
        )
        if not file_path:
            return

        self.track_path = Path(file_path)
        self.stem_paths.clear()
        self.mix_path = None
        self.stop_playback_players()
        self.position_slider.set_range(0, 0)
        self.elapsed_label.text = "0:00"
        self.duration_label.text = "0:00"
        self.set_playback_controls_enabled(False)
        self.set_mixer_content_visible(False)
        self.back_button.visible = True
        self.separate_button.visible = True
        self.separation_progress.visible = False
        self.track_label.text = self.track_path.name
        self.track_detail.text = str(self.track_path)
        self.separate_button.enabled = True
        self.status_bar().show_message(f"Selected {self.track_path.name}")


    def separate_stems(self) -> None:
        if not self.track_path:
            QMessageBox.warning(
                self, "No track selected", "Choose an audio file first."
            )
            return

        if self.worker_thread is not None:
            return

        self.separate_button.enabled = False
        self.upload_button.enabled = False
        self.separate_button.visible = False
        self.separation_progress.visible = True
        self.track_detail.text = "Separating with Demucs… this may take a few minutes."
        self.status_bar().show_message("Separating stems…")

        self.worker_thread = QThread(self)
        self.worker = SeparationWorker(self.track_path, self.output_dir)
        self.worker.move_to_thread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.separation_finished)
        self.worker.failed.connect(self.separation_failed)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.failed.connect(self.worker_thread.quit)
        self.worker_thread.finished.connect(self.thread_finished)
        self.worker_thread.start()

    @Slot()
    def thread_finished(self) -> None:
        if self.worker_thread:
            self.worker_thread.delete_later()

        self.worker_thread = None
        self.worker = None


    @Slot(dict)
    def separation_finished(self, stem_paths: dict[str, str]) -> None:
        self.stem_paths = stem_paths
        self.separation_progress.visible = False
        self.set_mixer_content_visible(True)
        self.upload_button.enabled = True
        self.separate_button.enabled = True
        self.track_detail.text = f"Extracted {len(stem_paths)} stems · adjust the controls below."
        self.status_bar().show_message("Stems ready")
        self.render_mix()


    @Slot(str)
    def separation_failed(self, message: str) -> None:
        self.separation_progress.visible = False
        self.set_mixer_content_visible(False)
        self.upload_button.enabled = True
        self.separate_button.enabled = True
        self.separate_button.visible = True
        self.track_detail.text = str(self.track_path)
        self.status_bar().show_message("Stem separation failed")
        QMessageBox.critical(
            self,
            "Stem separation failed",
            "Demucs could not process this track.\n\n" + message,
        )


    def schedule_mix(self) -> None:
        if self.stem_paths:
            self.mix_timer.start()


    def render_mix(self) -> None:
        if not self.stem_paths:
            return

        try:
            arrays: list[np.ndarray] = []
            sample_rate = 44100
            max_length = 0
            channels = 2

            for key in ("vocals", "drums", "bass", "guitar", "piano", "other"):
                path = self.stem_paths.get(key)
                if not path:
                    continue

                audio, sample_rate = sf.read(
                    path, always_2d=True, dtype="float32"
                )
                row = self.stem_rows[key]
                gain = (
                    0.0
                    if row.mute_button.checked
                    else row.slider.value / 100
                )
                arrays.append(audio * gain)
                max_length = max(max_length, len(audio))
                channels = max(channels, audio.shape[1])

            if not arrays:
                return

            mix = np.zeros((max_length, channels), dtype=np.float32)

            for audio in arrays:
                if audio.shape[1] == 1 and channels == 2:
                    audio = np.repeat(audio, 2, axis=1)

                mix[: len(audio), : audio.shape[1]] += audio

            mix = np.clip(mix, -1.0, 1.0)
            self.mix_path = self.output_dir / "current-mix.wav"
            sf.write(self.mix_path, mix, sample_rate)

            self.set_playback_controls_enabled(True)
            self.status_bar().show_message("Mix updated")

        except Exception as exc:
            self.status_bar().show_message(f"Could not update mix: {exc}")


    def set_mixer_content_visible(self, visible: bool) -> None:
        for widget in (
            self.stems_title,
            self.separate_button,
            self.stems_scroll,
            self.notice,
            self.footer,
            self.elapsed_label,
            self.position_slider,
            self.duration_label,
            self.play_button,
            self.pause_button,
            self.stop_button,
        ):
            widget.visible = visible


    def stem_gain(self, key: str) -> float:
        row = self.stem_rows[key]
        return 0.0 if row.mute_button.checked else row.slider.value / 100


    @Slot()
    def update_playback_volumes(self) -> None:
        for key, output in self.playback_outputs.items():
            output.set_volume(self.stem_gain(key))


    def stop_playback_players(self) -> None:
        for player in self.playback_players.values():
            player.stop()


    def ensure_playback_player(self, key: str, path: str) -> QMediaPlayer:
        player = self.playback_players.get(key)

        if player is None:
            player = QMediaPlayer(self)
            output = QAudioOutput(self)
            player.audio_output = output
            self.playback_players[key] = player
            self.playback_outputs[key] = output

            if self.playback_master is None:
                self.playback_master = player
                player.positionChanged.connect(self.update_playback_position)
                player.durationChanged.connect(self.update_playback_duration)

        source = QUrl.from_local_file(path)

        if player.source != source:
            player.set_source(source)

        self.playback_outputs[key].set_volume(self.stem_gain(key))
        return player


    def reset_to_upload(self) -> None:
        self.stop_playback_players()
        self.track_path = None
        self.stem_paths.clear()
        self.mix_path = None
        self.position_slider.set_range(0, 0)
        self.elapsed_label.text = "0:00"
        self.duration_label.text = "0:00"
        self.track_label.text = "No track selected"
        self.track_detail.text = "Supported formats: WAV, MP3, FLAC, M4A, OGG"
        self.separation_progress.visible = False
        self.set_mixer_content_visible(False)
        self.set_playback_controls_enabled(False)
        self.back_button.visible = False
        self.upload_button.enabled = True
        self.status_bar().show_message("Upload a track to get started")


    @staticmethod
    def format_duration(milliseconds: int) -> str:
        total_seconds = max(0, milliseconds // 1000)
        minutes, seconds = divmod(total_seconds, 60)
        return f"{minutes}:{seconds:02d}"


    def set_playback_controls_enabled(self, enabled: bool) -> None:
        self.play_button.enabled = enabled
        self.pause_button.enabled = enabled
        self.stop_button.enabled = enabled
        self.position_slider.enabled = enabled


    @Slot(int)
    def update_playback_position(self, position: int) -> None:
        self.position_slider.set_value(position)
        self.elapsed_label.text = self.format_duration(position)


    @Slot(int)
    def update_playback_duration(self, duration: int) -> None:
        self.position_slider.set_range(0, max(0, duration))
        self.duration_label.text = self.format_duration(duration)


    @Slot(int)
    def seek_mix(self, position: int) -> None:
        for player in self.playback_players.values():
            player.position = position


    def play_mix(self) -> None:
        if self.stem_paths:
            position = self.position_slider.value
            duration = self.position_slider.maximum

            if duration and position >= duration:
                position = 0

            players = []

            for key, path in self.stem_paths.items():
                player = self.ensure_playback_player(key, path)
                player.set_position(position)
                players.append(player)

            self.update_playback_volumes()

            for player in players:
                player.play()

        else:
            self.status_bar().show_message(
                "Choose a track and separate its stems before playing the mix."
            )


    def pause_mix(self) -> None:
        for player in self.playback_players.values():
            player.pause()


    def stop_mix(self) -> None:
        self.stop_playback_players()
        self.position_slider.set_value(0)


def main() -> int:
    app = QApplication(sys.argv)
    window = MixerWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
