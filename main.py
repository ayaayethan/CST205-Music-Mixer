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
        self.setObjectName("stemRow")
        self.setFixedHeight(76)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(18, 12, 18, 12)
        layout.setSpacing(16)

        name = QLabel(stem_name)
        name.setObjectName("stemName")
        name.setStyleSheet(f"color: {accent};")
        name.setFixedWidth(110)
        name.setMinimumHeight(26)

        self.slider = VolumeSlider(Qt.Orientation.Horizontal)
        self.slider.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.slider.setRange(0, 100)
        self.slider.setValue(100)
        self.slider.setAccessibleName(f"{stem_name} volume")
        self.slider.setMinimumSize(180, 32)

        self.volume_label = QLabel("100%")
        self.volume_label.setObjectName("volumeLabel")
        self.volume_label.setFixedWidth(56)
        self.volume_label.setMinimumHeight(26)
        self.slider.valueChanged.connect(
            lambda value: self.volume_label.setText(f"{value}%")
        )

        self.mute_button = QPushButton("MUTE")
        self.mute_button.setCheckable(True)
        self.mute_button.setMinimumSize(100, 38)
        self.mute_button.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.mute_button.setAccessibleName(f"Mute {stem_name}")
        self.mute_button.toggled.connect(
            lambda muted: self.mute_button.setText("UNMUTE" if muted else "MUTE")
        )

        layout.addWidget(name)
        layout.addWidget(self.slider, 1)
        layout.addWidget(self.volume_label)
        layout.addWidget(self.mute_button)


class MixerWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Stem Mixer")
        self.setMinimumSize(800, 650)
        self.resize(960, 900)
        self.track_path: Path | None = None
        self.stem_paths: dict[str, str] = {}
        self.output_dir = Path(tempfile.mkdtemp(prefix="stem-mixer-"))
        self.worker_thread: QThread | None = None
        self.worker: SeparationWorker | None = None
        self.stem_rows: dict[str, StemRow] = {}
        self.mix_path: Path | None = None
        self.mix_timer = QTimer(self)
        self.mix_timer.setSingleShot(True)
        self.mix_timer.setInterval(120)
        self.mix_timer.timeout.connect(self.render_mix)

        self.playback_players: dict[str, QMediaPlayer] = {}
        self.playback_outputs: dict[str, QAudioOutput] = {}
        self.playback_master: QMediaPlayer | None = None

        central_widget = QWidget()
        central_widget.setObjectName("app")
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(36, 30, 36, 24)
        layout.setSpacing(22)

        header = QHBoxLayout()
        title_block = QVBoxLayout()
        title = QLabel("STEM MIXER")
        title.setObjectName("title")
        subtitle = QLabel("Shape the balance of every layer.")
        subtitle.setObjectName("subtitle")
        title_block.addWidget(title)
        title_block.addWidget(subtitle)

        brand = QLabel("MIX / 01")
        brand.setObjectName("brand")
        header.addLayout(title_block)
        header.addStretch()
        self.back_button = QPushButton("Back to upload")
        self.back_button.setObjectName("secondaryButton")
        self.back_button.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.back_button.clicked.connect(self.reset_to_upload)
        self.back_button.hide()
        header.addWidget(self.back_button)
        header.addWidget(brand, alignment=Qt.AlignmentFlag.AlignTop)
        layout.addLayout(header)

        source_card = QFrame()
        source_card.setObjectName("card")
        source_layout = QVBoxLayout(source_card)
        source_layout.setContentsMargins(22, 18, 22, 18)
        source_layout.setSpacing(14)

        source_heading = QLabel("SOURCE TRACK")
        source_heading.setObjectName("sectionHeading")
        source_layout.addWidget(source_heading)

        source_row = QHBoxLayout()
        self.track_label = QLabel("No track selected")
        self.track_label.setObjectName("trackLabel")
        self.track_label.setMinimumWidth(0)
        self.track_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )

        self.upload_button = QPushButton("Choose audio file")
        self.upload_button.setObjectName("primaryButton")
        self.upload_button.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.upload_button.clicked.connect(self.choose_track)
        source_row.addWidget(self.track_label, 1)
        source_row.addWidget(self.upload_button)
        source_layout.addLayout(source_row)

        self.track_detail = QLabel(
            "Supported formats: WAV, MP3, FLAC, M4A, OGG"
        )
        self.track_detail.setObjectName("mutedText")
        source_layout.addWidget(self.track_detail)
        self.separation_progress = QProgressBar()
        self.separation_progress.setObjectName("separationProgress")
        self.separation_progress.setRange(0, 0)
        self.separation_progress.setTextVisible(False)
        self.separation_progress.setFixedHeight(7)
        self.separation_progress.hide()
        source_layout.addWidget(self.separation_progress)
        layout.addWidget(source_card)

        self.stems_header = QHBoxLayout()
        self.stems_title = QLabel("STEM CONTROLS")
        self.stems_title.setObjectName("sectionHeading")
        self.stems_header.addWidget(self.stems_title)
        self.stems_header.addStretch()
        self.separate_button = QPushButton("Separate stems")
        self.separate_button.setObjectName("secondaryButton")
        self.separate_button.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.separate_button.clicked.connect(self.separate_stems)
        self.stems_header.addWidget(self.separate_button)
        layout.addLayout(self.stems_header)

        stems_card = QFrame()
        stems_card.setObjectName("card")
        stems_layout = QVBoxLayout(stems_card)
        stems_layout.setContentsMargins(14, 8, 14, 8)
        stems_layout.setSpacing(8)

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
            stems_layout.addWidget(row)
        self.stems_scroll = QScrollArea()
        self.stems_scroll.setObjectName("stemsScroll")
        self.stems_scroll.setWidgetResizable(True)
        self.stems_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.stems_scroll.setMinimumHeight(200)
        self.stems_scroll.setWidget(stems_card)
        layout.addWidget(self.stems_scroll, 1)

        self.notice = QLabel("Choose a track, separate its stems, then adjust the mix.")
        self.notice.setObjectName("notice")
        self.notice.setWordWrap(True)
        layout.addWidget(self.notice)

        self.footer = QLabel("Upload a track to get started")
        self.footer.setObjectName("footer")
        layout.addWidget(self.footer)

        self.playback_seek_row = QHBoxLayout()
        self.elapsed_label = QLabel("0:00")
        self.elapsed_label.setObjectName("mutedText")
        self.elapsed_label.setFixedWidth(42)
        self.position_slider = QSlider(Qt.Orientation.Horizontal)
        self.position_slider.setObjectName("scrubber")
        self.position_slider.setRange(0, 0)
        self.position_slider.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.position_slider.setAccessibleName("Mix playback position")
        self.position_slider.sliderMoved.connect(self.seek_mix)
        self.duration_label = QLabel("0:00")
        self.duration_label.setObjectName("mutedText")
        self.duration_label.setFixedWidth(42)
        self.playback_seek_row.addWidget(self.elapsed_label)
        self.playback_seek_row.addWidget(self.position_slider, 1)
        self.playback_seek_row.addWidget(self.duration_label)
        layout.addLayout(self.playback_seek_row)

        self.playback_row = QHBoxLayout()
        self.playback_row.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.play_button = QPushButton()
        self.play_button.setObjectName("secondaryButton")
        self.play_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self.play_button.setIconSize(QSize(20, 20))
        self.play_button.setFixedSize(46, 40)
        self.play_button.setAccessibleName("Play mix")
        self.play_button.setToolTip("Play mix")
        self.play_button.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.play_button.clicked.connect(self.play_mix)
        self.pause_button = QPushButton()
        self.pause_button.setObjectName("secondaryButton")
        self.pause_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPause))
        self.pause_button.setIconSize(QSize(20, 20))
        self.pause_button.setFixedSize(46, 40)
        self.pause_button.setAccessibleName("Pause mix")
        self.pause_button.setToolTip("Pause mix")
        self.pause_button.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.pause_button.clicked.connect(self.pause_mix)
        self.stop_button = QPushButton()
        self.stop_button.setObjectName("secondaryButton")
        self.stop_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaStop))
        self.stop_button.setIconSize(QSize(20, 20))
        self.stop_button.setFixedSize(46, 40)
        self.stop_button.setAccessibleName("Stop mix")
        self.stop_button.setToolTip("Stop mix")
        self.stop_button.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.stop_button.clicked.connect(self.stop_mix)
        self.set_playback_controls_enabled(False)
        self.playback_row.addWidget(self.play_button)
        self.playback_row.addWidget(self.pause_button)
        self.playback_row.addWidget(self.stop_button)
        layout.addLayout(self.playback_row)

        self.set_mixer_content_visible(False)

        self.setStyleSheet(
            """
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
        )

    def choose_track(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
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
        self.position_slider.setRange(0, 0)
        self.elapsed_label.setText("0:00")
        self.duration_label.setText("0:00")
        self.set_playback_controls_enabled(False)
        self.set_mixer_content_visible(False)
        self.back_button.show()
        self.separate_button.show()
        self.separation_progress.hide()
        self.track_label.setText(self.track_path.name)
        self.track_detail.setText(str(self.track_path))
        self.separate_button.setEnabled(True)
        self.statusBar().showMessage(f"Selected {self.track_path.name}")

    def separate_stems(self) -> None:
        if not self.track_path:
            QMessageBox.warning(self, "No track selected", "Choose an audio file first.")
            return
        if self.worker_thread and self.worker_thread.isRunning():
            return

        self.separate_button.setEnabled(False)
        self.upload_button.setEnabled(False)
        self.separate_button.hide()
        self.separation_progress.show()
        self.track_detail.setText("Separating with Demucs… this may take a few minutes.")
        self.statusBar().showMessage("Separating stems…")

        self.worker_thread = QThread(self)
        self.worker = SeparationWorker(self.track_path, self.output_dir)
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.separation_finished)
        self.worker.failed.connect(self.separation_failed)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.failed.connect(self.worker_thread.quit)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        self.worker_thread.start()

    @Slot(dict)
    def separation_finished(self, stem_paths: dict[str, str]) -> None:
        self.stem_paths = stem_paths
        self.separation_progress.hide()
        self.set_mixer_content_visible(True)
        self.upload_button.setEnabled(True)
        self.separate_button.setEnabled(True)
        self.track_detail.setText(
            f"Extracted {len(stem_paths)} stems · adjust the controls below."
        )
        self.statusBar().showMessage("Stems ready")
        self.render_mix()

    @Slot(str)
    def separation_failed(self, message: str) -> None:
        self.separation_progress.hide()
        self.set_mixer_content_visible(False)
        self.upload_button.setEnabled(True)
        self.separate_button.setEnabled(True)
        self.separate_button.show()
        self.track_detail.setText(str(self.track_path))
        self.statusBar().showMessage("Stem separation failed")
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
                audio, sample_rate = sf.read(path, always_2d=True, dtype="float32")
                row = self.stem_rows[key]
                gain = 0.0 if row.mute_button.isChecked() else row.slider.value() / 100
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
            self.statusBar().showMessage("Mix updated")
        except Exception as exc:
            self.statusBar().showMessage(f"Could not update mix: {exc}")

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
            widget.setVisible(visible)

    def stem_gain(self, key: str) -> float:
        row = self.stem_rows[key]
        return 0.0 if row.mute_button.isChecked() else row.slider.value() / 100

    @Slot()
    def update_playback_volumes(self) -> None:
        for key, output in self.playback_outputs.items():
            output.setVolume(self.stem_gain(key))

    def stop_playback_players(self) -> None:
        for player in self.playback_players.values():
            player.stop()

    def ensure_playback_player(self, key: str, path: str) -> QMediaPlayer:
        player = self.playback_players.get(key)
        if player is None:
            player = QMediaPlayer(self)
            output = QAudioOutput(self)
            player.setAudioOutput(output)
            self.playback_players[key] = player
            self.playback_outputs[key] = output
            if self.playback_master is None:
                self.playback_master = player
                player.positionChanged.connect(self.update_playback_position)
                player.durationChanged.connect(self.update_playback_duration)

        source = QUrl.fromLocalFile(path)
        if player.source() != source:
            player.setSource(source)
        self.playback_outputs[key].setVolume(self.stem_gain(key))
        return player

    def reset_to_upload(self) -> None:
        self.stop_playback_players()
        self.track_path = None
        self.stem_paths.clear()
        self.mix_path = None
        self.position_slider.setRange(0, 0)
        self.elapsed_label.setText("0:00")
        self.duration_label.setText("0:00")
        self.track_label.setText("No track selected")
        self.track_detail.setText("Supported formats: WAV, MP3, FLAC, M4A, OGG")
        self.separation_progress.hide()
        self.set_mixer_content_visible(False)
        self.set_playback_controls_enabled(False)
        self.back_button.hide()
        self.upload_button.setEnabled(True)
        self.statusBar().showMessage("Upload a track to get started")

    @staticmethod
    def format_duration(milliseconds: int) -> str:
        total_seconds = max(0, milliseconds // 1000)
        minutes, seconds = divmod(total_seconds, 60)
        return f"{minutes}:{seconds:02d}"

    def set_playback_controls_enabled(self, enabled: bool) -> None:
        self.play_button.setEnabled(enabled)
        self.pause_button.setEnabled(enabled)
        self.stop_button.setEnabled(enabled)
        self.position_slider.setEnabled(enabled)

    @Slot(int)
    def update_playback_position(self, position: int) -> None:
        self.position_slider.setValue(position)
        self.elapsed_label.setText(self.format_duration(position))

    @Slot(int)
    def update_playback_duration(self, duration: int) -> None:
        self.position_slider.setRange(0, max(0, duration))
        self.duration_label.setText(self.format_duration(duration))

    @Slot(int)
    def seek_mix(self, position: int) -> None:
        for player in self.playback_players.values():
            player.setPosition(position)

    def play_mix(self) -> None:
        if self.stem_paths:
            position = self.position_slider.value()
            duration = self.position_slider.maximum()
            if duration and position >= duration:
                position = 0
            players = []
            for key, path in self.stem_paths.items():
                player = self.ensure_playback_player(key, path)
                player.setPosition(position)
                players.append(player)
            self.update_playback_volumes()
            for player in players:
                player.play()
        else:
            self.statusBar().showMessage(
                "Choose a track and separate its stems before playing the mix."
            )

    def pause_mix(self) -> None:
        for player in self.playback_players.values():
            player.pause()

    def stop_mix(self) -> None:
        self.stop_playback_players()
        self.position_slider.setValue(0)


def main() -> int:
    app = QApplication(sys.argv)
    window = MixerWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
