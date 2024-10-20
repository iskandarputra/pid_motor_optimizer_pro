# PID Motor Optimizer Pro

PID Motor Optimizer Pro is an advanced Python GUI application designed for precise control and optimization of motor systems. With an intuitive interface and powerful features, it empowers users to interact with motor control systems via serial communication, enabling real-time monitoring, PID parameter tuning, and automated optimization.

## Key Features

- **Precision Motor Control**: Easily start, stop, and adjust motor position using user-defined setpoints and timeouts.
- **Real-time PID Tuning**: Interactively fine-tune PID gains (Kp, Ki, Kd) and advanced settings to achieve optimal controller response.
- **Data Visualization**: Gain valuable insights into motor performance with interactive plots displaying position, setpoint, and error over time.
- **Automated Optimization**: Harness the power of a differential evolution algorithm to automatically find the best PID parameters, minimizing overshoot, settling time, and error.
- **Oscillation Safeguarding**: Ensure motor safety with real-time analysis of position data to detect and prevent harmful high-frequency oscillations during online optimization process.

## Python Requirements
- Python 3
- PyQt5, Matplotlib, NumPy, SciPy, pySerial

## STM32 Firmware

This repository hosts the Python GUI application for motor control and optimization. The corresponding STM32 firmware code is available upon request. 

If you're interested in acquiring the STM32 code or have any inquiries, please don't hesitate to reach out to me on LinkedIn or consider supporting the project by buying me a coffee. Your support is invaluable in maintaining and enhancing PID Motor Optimizer Pro.

Contact details:
- LinkedIn: https://www.linkedin.com/in/iskandarputra95/
- Buy Me a Coffee: TBA

I'm excited to connect with you and discuss the project in more detail!

## Getting Started
1. Clone the repository
2. Install dependencies
3. Configure the serial port and baud rate in `pid_motor_optimizer_pro.py` 
4. Launch the application: `python pid_motor_optimizer_pro.py`
5. Utilize the GUI to connect, control the motor, tune parameters, and perform optimizations

PID Motor Optimizer Pro combines the strength of Python and scientific computing libraries to provide a sophisticated yet user-friendly solution for precise motor control and optimization. Experience the difference it can make in your motor control projects.
