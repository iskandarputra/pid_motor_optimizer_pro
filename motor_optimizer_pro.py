#!/usr/bin/env python3

"""
This script provides a graphical user interface (GUI) for controlling and 
optimizing the performance of a motor connected to an STM32 microcontroller 
running Zephyr RTOS.

The GUI facilitates interaction with the motor control system through a 
serial connection, leveraging the Zephyr shell to send commands and receive data. 
This enables real-time monitoring and adjustment of the motor's behavior. A key 
feature of this application is its ability to detect and prevent damaging 
high-frequency oscillations in the motor during the optimization process.

**GUI Functionalities:**

* **Motor Control:**
    * **Start/Stop:** Initiate and terminate motor operation with specified setpoints and timeouts.
    * **Open/Close:**  Conveniently open or close the motor using predefined setpoints.
    * **Setpoint Control:** Define the desired target position for the motor.

* **PID Parameter Adjustment:**
    * **PID Gains:** Modify the proportional (Kp), integral (Ki), and derivative (Kd) gains 
      to fine-tune the controller's response.
    * **Output Limits:**  Set the minimum and maximum output values of the PID controller to 
      constrain the control signal.
    * **Setpoint Weight:** Adjust the influence of the setpoint on the control signal.
    * **Advanced Parameters:**  Configure additional parameters like deadband, setpoint ramp rate, 
      derivative filtering, and anti-windup gain to further refine the controller's behavior.

* **Real-time Data Visualization:**
    * **Interactive Plots:**  Observe the motor's position, setpoint, and error over time through 
      dynamically updated plots.
    * **Optimization Monitoring:**  Track the progress of PID parameter optimization, visualizing the 
      evolution of the best-fit cost and parameter values.

* **PID Parameter Optimization:**
    * **Automated Tuning:**  Employ a differential evolution algorithm to automatically determine 
      optimal PID parameters that minimize the cost function, which includes factors like 
      oscillation and settling time.
    * **Optimization Configuration:**  Define the parameter bounds and the number of iterations for 
      the optimization process.

* **Oscillation Detection and Prevention:**
    * **Real-time Analysis:**  Continuously analyzes the motor's position data to detect high-frequency 
      oscillations, which can occur when certain PID values are tested during optimization.
    * **Preventive Measures:**  If excessive oscillation is detected, the motor is immediately stopped to 
      prevent potential overheating or damage. This safeguards the motor during the optimization process.
    * **User Feedback:**  Provides clear warnings to the user through the GUI and log messages when 
      oscillations are detected.

**STM32 Firmware Features:**

The STM32 firmware, running on the Zephyr RTOS, provides the following capabilities:

* **Real-time PID Control:**  A dedicated thread executes the PID control algorithm at a high frequency, 
  ensuring a rapid response to changes in the motor's state.
* **PWM Output:**  Generates Pulse-Width Modulation (PWM) signals to control the motor's speed and direction.
* **Encoder Feedback:**  Reads the motor's position from an encoder, providing feedback to the PID controller.
* **Safety Mechanisms:**  Implements timeout and boundary checking to prevent unsafe operating conditions 
  and ensure reliable motor behavior.

This GUI application, in conjunction with the STM32 firmware, offers a comprehensive solution for 
precise motor control, automated PID parameter optimization, and oscillation management. It provides a 
user-friendly interface for monitoring, adjusting, and enhancing the performance of your motor control 
system while prioritizing safety and stability.

@author Iskandar Putra
@version 1.0
@date 2024-10-20
"""

import sys
import serial
import time
import logging
import re

import numpy as np
from scipy.optimize import differential_evolution

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QWidget, QLabel, QLineEdit, QTextEdit, 
                             QSpinBox, QStyleFactory, QFrame)
from PyQt5.QtCore import QTimer, QThread, pyqtSignal, Qt, QPropertyAnimation, QEasingCurve
from PyQt5.QtGui import QFont

# ===  Configurable Parameters ===
SERIAL_PORT = '/dev/ttyUSB0'
BAUD_RATE = 500000
UPDATE_INTERVAL = 1  # ms
MAX_DATA_POINTS = 500
OPTIMIZATION_BOUNDS = [(0.1, 2.0), (0.001, 0.1), (0.001, 0.05)]  # Bounds for Kp, Ki, Kd
EVALUATION_TIME = 0.9  # seconds for evaluating objective function
VIBRATION_WINDOW_SIZE = 10
VIBRATION_THRESHOLD = 8
# ===============================

COLORS = {
    'background': '#1D1D1D',
    'text': '#FFFFFF',
    'accent1': '#128BFF',  # Blue
    'accent2': '#01D903',  # Green
    'accent3': '#FFE92D',  # Yellow
    'accent4': '#FD4499',  # Pink
    'accent5': '#DF19FB',  # Purple
}

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class NeonButton(QPushButton):
    def __init__(self, text, color, parent=None):
        super().__init__(text, parent)
        self.setFont(QFont('Arial', 12, QFont.Bold))
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['background']};
                color: {color};
                border: 2px solid {color};
                border-radius: 10px;
                padding: 10px;
            }}
            QPushButton:hover {{
                background-color: {color};
                color: {COLORS['background']};
            }}
            QPushButton:pressed {{
                background-color: {COLORS['background']};
                color: {color};
            }}
        """)
        self.animation = QPropertyAnimation(self, b"geometry")
        self.animation.setDuration(100)
        self.animation.setEasingCurve(QEasingCurve.OutBounce)

    def enterEvent(self, event):
        self.animation.setStartValue(self.geometry())
        self.animation.setEndValue(self.geometry().adjusted(-2, -2, 2, 2))
        self.animation.start()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.animation.setStartValue(self.geometry())
        self.animation.setEndValue(self.geometry().adjusted(2, 2, -2, -2))
        self.animation.start()
        super().leaveEvent(event)


class NeonLineEdit(QLineEdit):
    def __init__(self, color, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"""
            QLineEdit {{
                background-color: {COLORS['background']};
                color: {color};
                border: 2px solid {color};
                border-radius: 5px;
                padding: 5px;
                font-size: 14px;
            }}
            QLineEdit:focus {{
                border: 2px solid {COLORS['accent5']};
            }}
        """)


class NeonSpinBox(QSpinBox):
    def __init__(self, color, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"""
            QSpinBox {{
                background-color: {COLORS['background']};
                color: {color};
                border: 2px solid {color};
                border-radius: 5px;
                padding: 5px;
                font-size: 14px;
            }}
            QSpinBox::up-button, QSpinBox::down-button {{
                background-color: {color};
                width: 20px;
            }}
            QSpinBox::up-arrow, QSpinBox::down-arrow {{
                width: 10px;
                height: 10px;
            }}
        """)


class OptimizationThread(QThread):
    """
    A QThread subclass for running the PID parameter optimization in a separate thread.

    This allows the optimization to run in the background without blocking the main UI thread.
    """
    iteration_complete = pyqtSignal(object, float, int)
    optimization_complete = pyqtSignal(object, float)

    def __init__(self, objective_function, bounds, maxiter):
        """
        Initialize the OptimizationThread.

        Args:
            objective_function (function): The function to minimize.
            bounds (list): The bounds of the parameters for the optimization.
            maxiter (int): The maximum number of iterations for the optimization.
        """
        super().__init__()
        self.objective_function = objective_function
        self.bounds = bounds
        self.maxiter = maxiter
        self.stop_flag = False

    def run(self):
        """
        Run the optimization using the differential evolution algorithm.
        """
        result = differential_evolution(
            self.objective_function,
            self.bounds,
            maxiter=self.maxiter,
            popsize=10,
            mutation=(0.5, 1),
            recombination=0.7,
            updating='deferred',
            workers=1,
            callback=self.callback
        )
        if not self.stop_flag:
            self.optimization_complete.emit(result.x, result.fun)

    def callback(self, xk, convergence):
        """
        Callback function for the differential evolution algorithm.

        Emits the `iteration_complete` signal to update the UI with the current parameters and cost.

        Args:
            xk (array): The current parameter values.
            convergence (float): The current convergence value.
        """
        if self.stop_flag:
            return True
        self.iteration_complete.emit(xk, self.objective_function(xk), 0)
        return False

    def stop(self):
        """
        Stop the optimization process.
        """
        self.stop_flag = True


class MainWindow(QMainWindow):
    """
    The main window of the application.

    Provides a GUI for controlling a motor with PID parameters, visualizing real-time data,
    and optimizing PID parameters.
    """
    def __init__(self):
        """
        Initialize the MainWindow.
        """
        super().__init__()
        self.setWindowTitle("PID Control with Optimization")
        self.setGeometry(100, 100, 1400, 900)
        self.setStyleSheet(f"background-color: {COLORS['background']}; color: {COLORS['text']};")
        
        # Initialize data arrays
        self.times = []
        self.setpoints = []
        self.positions = []
        self.errors = []
        
        # Data for optimization graphs
        self.iteration_numbers = []
        self.best_fit_costs = []
        self.kp_values = []
        self.ki_values = []
        self.kd_values = []
        
        # Initialize serial connection
        try:
            self.ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
            logger.info(f"Connected to {SERIAL_PORT}")
        except Exception as e:
            logger.error(f"Failed to open serial port: {str(e)}")
            sys.exit(1)
        
        self.setup_ui()
        self.setup_plot()
        
        # Optimization flag
        self.is_optimizing = False
        
        # Update timer
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update_plot_data)
        self.update_timer.start(UPDATE_INTERVAL)

    def setup_plot(self):
        """
        Set up the Matplotlib plot for visualizing data.
        """
        plt.style.use('dark_background')
        self.fig, ((self.ax1, self.ax2), (self.ax3, self.ax4)) = plt.subplots(2, 2, figsize=(12, 12))
        self.fig.patch.set_facecolor(COLORS['background'])
        
        self.line_setpoint, = self.ax1.plot([], [], label='Setpoint', color=COLORS['accent1'])
        self.line_position, = self.ax1.plot([], [], label='Position', color=COLORS['accent2'])
        self.line_error, = self.ax2.plot([], [], label='Error', color=COLORS['accent3'])
        
        self.line_best_fit, = self.ax3.plot([], [], label='Best Fit Cost', color=COLORS['accent4'])
        
        self.line_kp, = self.ax4.plot([], [], label='Kp', color=COLORS['accent1'])
        self.line_ki, = self.ax4.plot([], [], label='Ki', color=COLORS['accent2'])
        self.line_kd, = self.ax4.plot([], [], label='Kd', color=COLORS['accent3'])

        for ax in (self.ax1, self.ax2, self.ax3, self.ax4):
            ax.set_facecolor(COLORS['background'])
            ax.tick_params(colors=COLORS['text'])
            ax.xaxis.label.set_color(COLORS['text'])
            ax.yaxis.label.set_color(COLORS['text'])
            ax.title.set_color(COLORS['text'])
            ax.grid(color=COLORS['text'], linestyle='--', alpha=0.3)
        
        self.ax1.set_title('Position and Setpoint')
        self.ax2.set_title('Error')
        self.ax3.set_title('Optimization Progress')
        self.ax4.set_title('PID Parameters')
        
        self.ax3.set_xlabel('Iteration')
        self.ax3.set_ylabel('Best Fit Cost')
        self.ax4.set_xlabel('Iteration')
        self.ax4.set_ylabel('Parameter Value')
        
        for ax in (self.ax1, self.ax2, self.ax3, self.ax4):
            ax.legend(facecolor=COLORS['background'], edgecolor=COLORS['text'], labelcolor=COLORS['text'])
        
        self.fig.tight_layout()
        
        self.canvas = FigureCanvas(self.fig)
        self.plot_layout.addWidget(self.canvas)

    def setup_ui(self):
        """
        Set up the user interface elements.
        """
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        plot_panel = QFrame()
        plot_panel.setStyleSheet(f"background-color: {COLORS['background']}; border-radius: 10px;")
        self.plot_layout = QVBoxLayout(plot_panel)
        main_layout.addWidget(plot_panel, 7)
        
        control_panel = QFrame()
        control_panel.setStyleSheet(f"background-color: {COLORS['background']}; border-radius: 10px;")
        control_layout = QVBoxLayout(control_panel)
        main_layout.addWidget(control_panel, 3)
        
        buttons = [
            ('Start Control', self.start_motor_control, COLORS['accent1']),
            ('Stop Control', self.stop_motor_control, COLORS['accent2']),
            ('Open Motor', self.open_motor, COLORS['accent3']),
            ('Close Motor', self.close_motor, COLORS['accent4']),
            ('Clear Plot', self.clear_plot, COLORS['accent5']),
            ('Get Status', self.get_motor_control_status, COLORS['accent1'])
        ]
        
        for text, callback, color in buttons:
            button = NeonButton(text, color)
            button.clicked.connect(callback)
            control_layout.addWidget(button)
        
        pid_layout = QHBoxLayout()
        pid_layout.addWidget(QLabel("PID:"))
        self.kp_entry = NeonLineEdit(COLORS['accent1'])
        self.ki_entry = NeonLineEdit(COLORS['accent2'])
        self.kd_entry = NeonLineEdit(COLORS['accent3'])
        for entry in (self.kp_entry, self.ki_entry, self.kd_entry):
            pid_layout.addWidget(entry)
        self.pid_button = NeonButton("Set PID", COLORS['accent4'])
        self.pid_button.clicked.connect(self.set_pid)
        pid_layout.addWidget(self.pid_button)
        control_layout.addLayout(pid_layout)
        
        setpoint_layout = QHBoxLayout()
        setpoint_layout.addWidget(QLabel("Setpoint:"))
        self.setpoint_entry = NeonLineEdit(COLORS['accent5'])
        setpoint_layout.addWidget(self.setpoint_entry)
        self.setpoint_button = NeonButton("Set Setpoint", COLORS['accent1'])
        self.setpoint_button.clicked.connect(self.set_setpoint)
        setpoint_layout.addWidget(self.setpoint_button)
        control_layout.addLayout(setpoint_layout)
        
        optimization_layout = QVBoxLayout()
        self.start_optimization_button = NeonButton("Start Optimization", COLORS['accent2'])
        self.stop_optimization_button = NeonButton("Stop Optimization", COLORS['accent3'])
        self.start_optimization_button.clicked.connect(self.start_optimization)
        self.stop_optimization_button.clicked.connect(self.stop_optimization)
        self.stop_optimization_button.setEnabled(False)
        optimization_layout.addWidget(self.start_optimization_button)
        optimization_layout.addWidget(self.stop_optimization_button)
        
        iteration_layout = QHBoxLayout()
        iteration_layout.addWidget(QLabel("Iterations:"))
        self.iteration_spinner = NeonSpinBox(COLORS['accent4'])
        self.iteration_spinner.setRange(1, 100)
        self.iteration_spinner.setValue(50)
        iteration_layout.addWidget(self.iteration_spinner)
        optimization_layout.addLayout(iteration_layout)
        
        control_layout.addLayout(optimization_layout)
        
        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setStyleSheet(f"""
            QTextEdit {{
                background-color: {COLORS['background']};
                color: {COLORS['text']};
                border: 2px solid {COLORS['accent5']};
                border-radius: 10px;
                padding: 10px;
                font-family: 'Courier New';
                font-size: 12px;
            }}
        """)
        control_layout.addWidget(self.status_text)

    def update_plot_data(self):
        """
        Update the plot with the latest data received from the motor.
        """
        while self.ser.in_waiting:
            line = self.ser.readline().decode('utf-8').strip()
            data = self.parse_serial_data(line)
            if data:
                time, setpoint, position, error = data
                self.times.append(time)
                self.setpoints.append(setpoint)
                self.positions.append(position)
                self.errors.append(error)

                # Limit the number of data points
                if len(self.times) > MAX_DATA_POINTS:
                    self.times = self.times[-MAX_DATA_POINTS:]
                    self.setpoints = self.setpoints[-MAX_DATA_POINTS:]
                    self.positions = self.positions[-MAX_DATA_POINTS:]
                    self.errors = self.errors[-MAX_DATA_POINTS:]

                # Check for vibration
                if self.detect_vibration(VIBRATION_WINDOW_SIZE, VIBRATION_THRESHOLD):
                    logger.warning("Vibration detected!")
                    self.status_text.append("Warning: Vibration detected!")
                    self.send_motor_control_command("motor_control stop_immediate")

        self.line_setpoint.set_data(self.times, self.setpoints)
        self.line_position.set_data(self.times, self.positions)
        self.line_error.set_data(self.times, self.errors)

        for ax in (self.ax1, self.ax2):
            ax.relim()
            ax.autoscale_view()

        self.canvas.draw()

    def parse_serial_data(self, line):
        """
        Parse the serial data received from the motor.

        Args:
            line (str): The line of serial data.

        Returns:
            list: A list containing the parsed time, setpoint, position, and error values.
                  Returns None if the line does not match the expected pattern.
        """
        pattern = r"Time: ([\d.]+), Setpoint: ([\d.]+), Position: ([-\d.]+), Error: ([-\d.]+), Control: ([\d.]+)"
        match = re.search(pattern, line)
        if match:
            return [float(match.group(1)), float(match.group(2)), float(match.group(3)), float(match.group(4))]
        return None

    def send_motor_control_command(self, command):
        """
        Send a command to the motor through the serial port.

        Args:
            command (str): The command to send.
        """
        try:
            self.ser.write(f"{command}\r\n".encode('utf-32'))
            logger.info(f"Sent: {command}")
            self.status_text.append(f"Sent: {command}")
        except Exception as e:
            logger.error(f"Error sending command: {e}")
            self.status_text.append(f"Error sending command: {e}")

    def start_motor_control(self):
        """
        Start the motor control process.
        """
        self.clear_plot()
        time.sleep(0.1)
        self.send_motor_control_command("motor_control start 1000 1000") # setpoint and timeout

    def stop_motor_control(self):
        """
        Stop the motor control process.
        """
        self.send_motor_control_command("motor_control stop")

    def open_motor(self):
        self.send_motor_control_command("motor_control open")

    def close_motor(self):
        self.send_motor_control_command("motor_control close")

    def clear_plot(self):
        """
        Clear the plot.
        """
        self.times.clear()
        self.setpoints.clear()
        self.positions.clear()
        self.errors.clear()
        
        for line in (self.line_setpoint, self.line_position, self.line_error):
            line.set_data([], [])
        
        for ax in (self.ax1, self.ax2):
            ax.relim()
            ax.autoscale_view()
        
        self.canvas.draw()
        self.canvas.flush_events()

    def clear_optimization_plots(self):
        """
        Clear the optimization plots.
        """
        self.iteration_numbers.clear()
        self.best_fit_costs.clear()
        self.kp_values.clear()
        self.ki_values.clear()
        self.kd_values.clear()
        
        self.line_best_fit.set_data([], [])
        self.line_kp.set_data([], [])
        self.line_ki.set_data([], [])
        self.line_kd.set_data([], [])
        
        for ax in (self.ax3, self.ax4):
            ax.relim()
            ax.autoscale_view()
        
        self.canvas.draw()
        self.canvas.flush_events()

    def set_pid(self):
        """
        Set the PID parameters for the motor control.
        """
        try:
            kp = float(self.kp_entry.text())
            ki = float(self.ki_entry.text())
            kd = float(self.kd_entry.text())
            self.send_motor_control_command(f"motor_control set_pid {kp} {ki} {kd}")
        except ValueError:
            logger.error("Invalid PID values entered")
            self.status_text.append("Error: Invalid PID values. Please enter valid numbers.")

    def set_setpoint(self):
        """
        Set the setpoint for the motor control.
        """
        try:
            setpoint = float(self.setpoint_entry.text())
            self.send_motor_control_command(f"motor_control start {setpoint} 5000")  # 5000ms timeout
        except ValueError:
            logger.error("Invalid setpoint value entered")
            self.status_text.append("Error: Invalid setpoint value. Please enter a valid number.")

    def get_motor_control_status(self):
        """
        Get and display the motor control status.
        """
        self.send_motor_control_command("motor_control status")
        # Wait for response
        time.sleep(0.5)
        response = ""
        timeout = time.time() + 2.0
        while time.time() < timeout:
            if self.ser.in_waiting:
                line = self.ser.readline().decode('utf-8').strip()
                response += line + "\n"
                self.status_text.append(line)
            else:
                time.sleep(0.1)
        if not response:
            logger.warning("No status response received")
            self.status_text.append("No status response received.")

    def start_optimization(self):
        """
        Start the PID parameter optimization process.
        """
        self.clear_optimization_plots()
        self.is_optimizing = True
        self.start_optimization_button.setEnabled(False)
        self.stop_optimization_button.setEnabled(True)

        self.optimization_thread = OptimizationThread(
            self.objective_function,
            OPTIMIZATION_BOUNDS, 
            maxiter=self.iteration_spinner.value()
        )
        self.optimization_thread.iteration_complete.connect(self.on_iteration_complete)
        self.optimization_thread.optimization_complete.connect(self.on_optimization_complete)
        self.optimization_thread.start()

    def stop_optimization(self):
        """
        Stop the PID parameter optimization process.
        """
        if hasattr(self, 'optimization_thread'):
            self.optimization_thread.stop()
            self.optimization_thread.wait()
        self.is_optimizing = False
        self.start_optimization_button.setEnabled(True)
        self.stop_optimization_button.setEnabled(False)
        logger.info("Optimization stopped")

    def detect_vibration(self, window_size, threshold):
        """
        Detect vibration in the motor's position data.

        Args:
            window_size (int): The size of the window to analyze for vibration.
            threshold (int): The threshold for the number of direction changes to consider as vibration.

        Returns:
            bool: True if vibration is detected, False otherwise.
        """
        if len(self.positions) < window_size:
            return False
        
        recent_positions = self.positions[-window_size:]
        direction_changes = 0
        last_direction = 0
        
        for i in range(1, len(recent_positions)):
            current_direction = 1 if recent_positions[i] > recent_positions[i-1] else -1
            if current_direction != last_direction and last_direction != 0:
                direction_changes += 1
            last_direction = current_direction
        
        return direction_changes >= threshold

    def objective_function(self, params):
        """
        Objective function for the optimization algorithm.

        Evaluates the performance of the motor control with the given PID parameters.

        Args:
            params (list): The PID parameters to evaluate (Kp, Ki, Kd).

        Returns:
            float: The cost associated with the evaluated parameters.
        """
        if not self.is_optimizing:
            return float('inf')

        kp, ki, kd = params
        time.sleep(0.5)
        self.clear_plot()
        self.send_motor_control_command(f"motor_control set_pid {kp} {ki} {kd}")
        time.sleep(0.1)
        self.send_motor_control_command("motor_control start 1000 1000")  
        time.sleep(0.5)

        start_time = time.time()

        while time.time() - start_time < EVALUATION_TIME and self.is_optimizing:
            if self.detect_vibration(VIBRATION_WINDOW_SIZE, VIBRATION_THRESHOLD):
                logger.warning("Vibration detected during optimization. Stopping motor.")
                self.send_motor_control_command("motor_control stop_immediate")
                return float('inf') 
            time.sleep(UPDATE_INTERVAL / 1000)

        self.send_motor_control_command("motor_control stop")

        if not self.is_optimizing or not self.errors:
            return float('inf')

        mean_error = np.mean(self.errors[-100:]) 

        # Calculate overshoot
        overshoot = 0
        if self.positions and self.setpoints:
            target = self.setpoints[-1] 
            max_position = max(self.positions)
            try:
                overshoot = max(0, (max_position - target) / target) * 100 
            except ZeroDivisionError:
                return float('inf')  

        # Calculate settling time (time to get within 2% of final value)
        settling_time = EVALUATION_TIME
        if self.positions and self.setpoints:
            target = self.setpoints[-1]
            tolerance = 0.02 * target  # 2% of target
            for i, pos in enumerate(self.positions):
                if abs(pos - target) <= tolerance:
                    settling_time = i * (UPDATE_INTERVAL / 1000)
                    break

        # Combine metrics
        cost = mean_error + 0.1 * overshoot + 0.1 * settling_time

        logger.info(f"Evaluation - Kp: {kp:.4f}, Ki: {ki:.4f}, Kd: {kd:.4f}, "
                    f"Mean Error: {mean_error:.4f}, Overshoot: {overshoot:.2f}%, "
                    f"Settling Time: {settling_time:.4f}s, Cost: {cost:.4f}")

        return cost

    def read_latest_data(self):
        """
        Read the latest data from the motor.

        Returns:
            dict: A dictionary containing the latest error, position, and setpoint values.
                  Returns None if no data is available.
        """
        if self.errors and self.positions and self.setpoints:
            return {
                'error': self.errors[-1],
                'position': self.positions[-1],
                'setpoint': self.setpoints[-1]
            }
        return None

    def on_iteration_complete(self, params, cost, iteration):
        """
        Handle the completion of an iteration in the optimization process.

        Args:
            params (list): The PID parameters evaluated in the iteration.
            cost (float): The cost associated with the evaluated parameters.
            iteration (int): The iteration number.
        """
        kp, ki, kd = params
        self.iteration_numbers.append(len(self.iteration_numbers) + 1)
        self.best_fit_costs.append(cost)
        self.kp_values.append(kp)
        self.ki_values.append(ki)
        self.kd_values.append(kd)
        
        self.update_optimization_plots()
        
        logger.info(f"Iteration {len(self.iteration_numbers)} complete: Kp={kp:.4f}, Ki={ki:.4f}, Kd={kd:.4f}, Cost={cost:.4f}")
        self.status_text.append(f"Iteration {len(self.iteration_numbers)} complete: Kp={kp:.4f}, Ki={ki:.4f}, Kd={kd:.4f}, Cost={cost:.4f}")

    def update_optimization_plots(self):
        """
        Update the optimization plots with the latest data.
        """
        self.line_best_fit.set_data(self.iteration_numbers, self.best_fit_costs)
        self.line_kp.set_data(self.iteration_numbers, self.kp_values)
        self.line_ki.set_data(self.iteration_numbers, self.ki_values)
        self.line_kd.set_data(self.iteration_numbers, self.kd_values)
        
        for ax in (self.ax3, self.ax4):
            ax.relim()
            ax.autoscale_view()
        
        self.canvas.draw()
        QApplication.processEvents()

    def on_optimization_complete(self, best_params, best_score):
        """
        Handle the completion of the optimization process.

        Args:
            best_params (list): The best PID parameters found by the optimization algorithm.
            best_score (float): The cost associated with the best parameters.
        """
        kp, ki, kd = best_params
        logger.info(f"Optimization complete: Kp={kp:.4f}, Ki={ki:.4f}, Kd={kd:.4f}, Score={best_score:.4f}")
        self.status_text.append(f"Optimization complete: Kp={kp:.4f}, Ki={ki:.4f}, Kd={kd:.4f}, Score={best_score:.4f}")
        self.kp_entry.setText(f"{kp:.4f}")
        self.ki_entry.setText(f"{ki:.4f}")
        self.kd_entry.setText(f"{kd:.4f}")
        self.start_optimization_button.setEnabled(True)
        self.stop_optimization_button.setEnabled(False)
        self.is_optimizing = False

    def closeEvent(self, event):
        """
        Handle the window close event.

        Closes the serial connection and stops the optimization thread if it is running.
        """
        if hasattr(self, 'ser') and self.ser.is_open:
            self.ser.close()
        if hasattr(self, 'optimization_thread'):
            self.optimization_thread.stop()
            self.optimization_thread.wait()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create('Fusion'))
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())