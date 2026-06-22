import os, sys
import markdown
from PyQt6.QtWidgets import(
    QWidget, QPushButton, QHBoxLayout,
    QVBoxLayout, QProgressDialog,
    QStackedWidget, QSizePolicy, QMessageBox, QTextBrowser, QButtonGroup
) 
from PyQt6.QtCore import Qt, QSize, QThread
from PyQt6.QtGui import QIcon

from application.analysis_window import AnalysisWindow
from application.worker import AnalysisWorker, StartupWorker
from application.recon_window import ReconWindow
from application.prediction_dialog import PredictionDialog
from application.logger import logger, log_emitter
from application.log_window import LogWindow

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.thread = None
        self.worker = None
        self.backend_ready = False
        self.analysis_stopping = False

        log_emitter.log_signal.connect(self.test_log)

        self.stack = QStackedWidget()

        self.recon_page = ReconWindow()
        self.recon_page.analyze_requested.connect(self.handle_analysis)

        self.setWindowTitle("QuantPhase")
        self.setGeometry(100, 100, 1000, 650)

        self.main_layout = QHBoxLayout()
        self.setLayout(self.main_layout)

        # about
        self.page_about = QWidget()
        about_layout = QVBoxLayout()

        self.about_browser = QTextBrowser()
        self.about_browser.setOpenExternalLinks(True)

        def resource_path(relative_path):
            try:
                base_path = sys._MEIPASS
            except Exception:
                base_path = os.path.abspath(".")
            return os.path.join(base_path, relative_path)

        with open(resource_path("application/about.md"), "r", encoding="utf-8") as f:
            md_text = f.read()

        html = markdown.markdown(md_text)

        # (optional) add some basic styling
        styled_html = f"""
            <html>
            <head>
            <style>
                body {{
                    font-family: 'Segoe UI', Arial;
                    line-height: 1.3;
                    color: #e0e0e0;
                    background-color: #121212;
                    font-size: 15px;
                }}
                .container {{
                    margin-left: 100px;
                    margin-right: 20px;
                    margin-top: 20px;
                    margin-bottom: 20px;
                    max-width: 800px;
                }}
                h1 {{
                    color: #ffffff;
                    margin-bottom: 10px;
                }}
                h2 {{
                    color: #cccccc;
                    margin-top: 20px;
                }}
                p {{
                    margin-bottom: 10px;
                }}
                li {{
                    margin-bottom: 6px;
                }}
            </style>
            </head>
            <body>

            <table width="100%">

            <tr>
            <td height="30"></td>
            </tr>

            <tr>
            <td width="40"></td>

            <td>
            {html}
            </td>

            <td width="40"></td>
            </tr>

            <tr>
            <td height="30"></td>
            </tr>

            <td></td>
            </tr>
            </table>

            </body>
            </html>
            """

        self.about_browser.setHtml(styled_html)

        about_layout.addWidget(self.about_browser)
        self.page_about.setLayout(about_layout)

        # Sidebar container
        self.sidebar_widget = QWidget()
        self.sidebar_layout = QVBoxLayout()
        self.sidebar_widget.setLayout(self.sidebar_layout)

        self.sidebar_expanded = True

        # Collapse button
        self.toggle_btn = QPushButton("☰")
        self.toggle_btn.setFixedHeight(40)
        self.toggle_btn.clicked.connect(self.toggle_sidebar)

        # Navigation buttons
        self.btn_about = QPushButton("About")
        self.btn_recon = QPushButton("Reconstruction")
        self.btn_analysis = QPushButton("Analysis")
        self.btn_logs = QPushButton("Logs")

        self.sidebar_widget.setObjectName("sidebar")

        self.btn_about.setIcon(QIcon(resource_path("application/assets/icons/info.svg")))
        self.btn_recon.setIcon(QIcon(resource_path("application/assets/icons/build.svg")))
        self.btn_analysis.setIcon(QIcon(resource_path("application/assets/icons/analytics.svg")))
        self.btn_logs.setIcon(QIcon(resource_path("application/assets/icons/terminal.svg")))

        sidebar_style = """
        #sidebar {
            background-color: #1e1e1e;
        }

        #sidebar QPushButton {
            border: none;
            background-color: transparent;
            color: #dddddd;
            padding: 8px 12px;
            text-align: left;
            font-size: 14px;
        }

        #sidebar QPushButton:hover {
            background-color: #2a2a2a;
        }

        #sidebar QPushButton:checked {
            background-color: #3a3a3a;
            color: #ffffff;
            font-weight: bold;
        }
        """
        self.sidebar_widget.setStyleSheet(sidebar_style)

        self.nav_buttons = [
            self.btn_about,
            self.btn_recon,
            self.btn_analysis,
            self.btn_logs
        ]

        self.button_group = QButtonGroup()
        self.button_group.setExclusive(True)

        for btn in self.nav_buttons:
            self.button_group.addButton(btn)

        for btn in self.nav_buttons:
            btn.setCheckable(True)

        for btn in self.nav_buttons:
            btn.setFixedHeight(40)
            btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        for btn in self.nav_buttons:
            btn.setStyleSheet("text-align: left; padding-left: 10px;")

        # Add to sidebar
        self.sidebar_layout.addWidget(self.toggle_btn)
        for btn in self.nav_buttons:
            self.sidebar_layout.addWidget(btn)

        for btn in self.nav_buttons:
            btn.setIconSize(QSize(20, 20))

        self.sidebar_layout.addStretch()
        self.sidebar_widget.setFixedWidth(180)

        self.analysis_page = AnalysisWindow()
        self.log_page = LogWindow()

        self.stack.addWidget(self.page_about)     # index 0
        self.stack.addWidget(self.recon_page)      # index 1
        self.stack.addWidget(self.analysis_page)   # index 2
        self.stack.addWidget(self.log_page)     # index 3

        self.stack.setCurrentIndex(0)
        self.btn_about.setChecked(True)

        self.btn_about.clicked.connect(self.open_about_page)
        self.btn_recon.clicked.connect(self.open_recon_page)
        self.btn_analysis.pressed.connect(self.open_analysis_page)
        self.btn_logs.clicked.connect(self.open_logs_page)

        self.main_layout.addWidget(self.sidebar_widget)
        self.main_layout.addWidget(self.stack)

        # Startup Worker
        self.start_startup_worker()

    def open_logs_page(self):

        self.stack.setCurrentIndex(3)

        self.set_active_button(self.btn_logs)

    def test_log(self, message):
        print("GUI RECEIVED:", message)

    def start_startup_worker(self):

        self.startup_thread = QThread()
        self.startup_worker = StartupWorker()

        self.startup_worker.moveToThread(self.startup_thread)

        self.startup_thread.started.connect(
            self.startup_worker.run
        )

        self.startup_worker.finished.connect(
            self.on_startup_finished
        )

        self.startup_worker.log.connect(
            self.log_message
        )

        self.startup_worker.finished.connect(
            self.startup_thread.quit
        )

        self.startup_worker.finished.connect(
            self.startup_worker.deleteLater
        )

        self.startup_thread.finished.connect(
            self.startup_thread.deleteLater
        )

        self.startup_thread.start()


    def log_message(self, message):
        logger.info(message)


    def on_startup_finished(self):

        self.backend_ready = True

        logger.info("Backend ready")

    def set_active_button(self, active_btn):
        active_btn.setChecked(True)

    def toggle_sidebar(self):
        if self.sidebar_expanded:
            self.sidebar_widget.setFixedWidth(60)

            # Remove text (icons only)
            self.btn_about.setText("")
            self.btn_recon.setText("")
            self.btn_analysis.setText("")
            self.btn_logs.setText("")

            for btn in self.nav_buttons:
                btn.setStyleSheet("text-align: center;")

        else:
            self.sidebar_widget.setFixedWidth(180)

            # Restore text
            self.btn_about.setText("About")
            self.btn_recon.setText("Reconstruction")
            self.btn_analysis.setText("Analysis")
            self.btn_logs.setText("Logs")

            for btn in self.nav_buttons:
                btn.setStyleSheet("text-align: left; padding-left: 10px;")

        self.sidebar_expanded = not self.sidebar_expanded
    
    def open_about_page(self):
        self.stack.setCurrentIndex(0)
        self.set_active_button(self.btn_about)

    def open_recon_page(self):
        self.stack.setCurrentIndex(1)
        self.set_active_button(self.btn_recon)

    def show_reconstruction(self):
        print("Switched to Reconstruction view")

    def show_analysis(self):
        print("Switched to Analysis view")

    def handle_analysis(self, phase):

        if self.analysis_stopping:
            return

        if self.thread is not None:
            try:
                if self.thread.isRunning():
                    return
            except RuntimeError:
                self.thread = None

        self.progress = QProgressDialog(
            "Analyzing...",
            None,
            0,
            100,
            self
        )

        self.progress.setWindowTitle("Analysis")
        self.progress.setMinimumDuration(0)
        self.progress.setAutoClose(False)
        self.progress.setAutoReset(False)
        self.progress.setWindowModality(Qt.WindowModality.WindowModal)
        self.progress.show()
        self.progress.canceled.connect(self.cancel_analysis)

        self.thread = QThread()
        self.worker = AnalysisWorker(phase)

        self.worker.moveToThread(self.thread)

        # Start
        self.thread.started.connect(self.worker.run)

        # Progress
        self.worker.progress.connect(self.update_progress)

        # Done
        self.worker.finished.connect(self.on_analysis_finished)
        self.worker.error.connect(self.on_analysis_error)

        # Cleanup
        self.worker.finished.connect(self.thread.quit)
        self.worker.error.connect(self.thread.quit)

        self.worker.finished.connect(self.worker.deleteLater)

        self.thread.finished.connect(self.cleanup_analysis_thread)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()

    def cancel_analysis(self):

        logger.info("Stopping analysis process...")

        self.analysis_stopping = True

        if self.worker is not None:
            self.worker.cancel()

        if self.progress is not None:
            self.progress.close()

        # Disable analyze button temporarily
        self.recon_page.analyze_button.setEnabled(False)
        self.recon_page.analyze_button.setText("Stopping...")
    
    def cleanup_analysis_thread(self):

        if self.thread is None:
            return
        
        if self.thread is not None:
            self.thread.deleteLater()

        self.thread = None
        self.worker = None

        logger.info("Analysis process stopped")

        self.analysis_stopping = False

        self.recon_page.analyze_button.setText("Analyze")

        if self.recon_page.selected_index is not None:
            self.recon_page.analyze_button.setEnabled(True)

    def open_analysis_page(self):
        if hasattr(self, "latest_table_data"):
            self.stack.setCurrentIndex(2)
            self.set_active_button(self.btn_analysis)
        else:
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Icon.Warning)
            msg.setWindowTitle("Analysis Not Available")
            msg.setText("No analysis results available yet.")
            msg.setInformativeText("Please generate reconstructions and click Analyze first.")
            msg.exec()

    def run_prediction(self):
        from backend.pipeline import predict_from_table

        if not self.backend_ready:
            return

        if not hasattr(self, "latest_table_data"):
            return

        try:
            prediction, confidence = predict_from_table(self.latest_table_data)

            dialog = PredictionDialog(prediction, confidence, self)
            dialog.exec()

        except Exception as e:
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Icon.Critical)
            msg.setWindowTitle("Prediction Error")
            msg.setText("Prediction failed")
            msg.setInformativeText(str(e))
            msg.exec()
        
    def update_progress(self, value, text):
        self.progress.setValue(value)
        self.progress.setLabelText(text)
        
    def on_analysis_finished(self, segmented_image, table_data, stats_data):

        self.progress.close()

        # Create new analysis page
        new_page = AnalysisWindow(
            segmented_image=segmented_image,
            table_data=table_data,
            stats_data=stats_data
        )

        # Replace old page in stack
        old_widget = self.stack.widget(2)
        self.stack.removeWidget(old_widget)
        old_widget.deleteLater()

        self.stack.insertWidget(2, new_page)
        self.analysis_page = new_page

        # Store data
        self.latest_table_data = table_data
        self.latest_segmented_image = segmented_image

        # Connect prediction button
        # self.analysis_page.predict_button.clicked.connect(self.run_prediction)

        # Switch view
        if self.stack.currentIndex() == 1:
            self.stack.setCurrentIndex(2)
        self.set_active_button(self.btn_analysis)

    def on_analysis_error(self, error_msg):
        if self.progress is not None:
            self.progress.close()

        self.cleanup_analysis_thread()
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Icon.Critical)
        msg.setWindowTitle("Analysis Failed")
        msg.setText("Something went wrong during analysis.")
        msg.setInformativeText(error_msg)
        msg.exec()