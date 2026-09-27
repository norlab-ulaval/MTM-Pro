# Quadcopter datasets

## PI-TCN

- [Dataset](https://github.com/arplaboratory/pi-tcn)
- [Data post-processing script](https://github.com/arplaboratory/PI-TCN/blob/main/process_data.py)

![table_1_pi_tcn.png](data/external_data/long_horizon/image/table_1_pi_tcn.png)

### Notes

- 68 trajectories
- ~58 min of flight time
- linear velocity is in the inertial frame ⚠️
- angular velocity is in the body frame

### Post-processing

- syncronizes selected topics
- python script option to filter the parsed data to minimize the effect of sensor noise (low-pass Butterworth and
  Savitzky–Golay filters)

### CSV header:

Vicom: Captured at 100hz
IMU: Captured at ?
Motor: Captured at ?

| Quantity                           | Header Abbreviation | Frame    | Source | Post-process                              | Notes                             |
|------------------------------------|---------------------|----------|--------|-------------------------------------------|-----------------------------------|
| time [s]                           | t                   |          | Vicom  |                                           |                                   |
| position x [m]                     | p_x                 |          | Vicom  |                                           |                                   |
| position y [m]                     | p_y                 |          | ...    |                                           |                                   |
| position z [m]                     | p_z                 |          | ...    |                                           |                                   |
| quaternion qx                      | q_w                 |          | Vicom  | filtered using a 4th order                |                                   |
| quaternion qy                      | q_x                 |          | ...    | Butterworth lowpass filter                |                                   |
| quaternion qz                      | q_y                 |          | ...    | with a cutoff frequency of 5              |                                   |
| quaternion qw                      | q_z                 |          | ...    | ...                                       |                                   |
| linear velocity x [m/s]            | v_x                 | inertial |        |                                           | not the same frame as Neurobem ⚠️ |
| linear velocity y [m/s]            | v_y                 | ...      |        |                                           |                                   |
| linear velocity z [m/s]            | v_z                 | ...      |        |                                           |                                   |
| linear acceleration x [m/s^2]      | vdot_x              | body     |        | recover from velocity, filtered using UKF |                                   |
| linear acceleration y [m/s^2]      | vdot_y              | ...      |        | ...                                       |                                   |
| linear acceleration z [m/s^2]      | vdot_z              | ...      |        | ...                                       |                                   |
| angular velocity x [rad/s]         | w_x                 | body     | IMU    |                                           |                                   |
| angular velocity y [rad/s]         | w_y                 | ...      | ...    |                                           |                                   |
| angular velocity z [rad/s]         | w_z                 | ...      | ...    |                                           |                                   |
| angular acceleration x [rad/s^2]   | wdot_x              | body     |        | recover from velocity, filtered using UKF |                                   |
| angular acceleration y [rad/s^2]   | wdot_y              | ...      |        | ...                                       |                                   |
| angular acceleration z [rad/s^2]   | wdot_z              | ...      |        | ...                                       |                                   |
| motor speed back right [rad/s]     | u_0                 |          | ESC    | scaled by 0.001                           |                                   |
| motor speed front right [rad/s]    | u_1                 |          | ...    | and filtered using a 4th order            |                                   |
| motor speed back left  [rad/s]     | u_2                 |          | ...    | Butterworth lowpass filter                |                                   |
| motor speed front left [rad/s]     | u_3                 |          | ...    | with a cutoff frequency of 5              |                                   |
| derivative motor speed 1 [rad/s^2] | null                |          |        |                                           |                                   |
| derivative motor speed 2 [rad/s^2] | null                |          |        |                                           |                                   |
| derivative motor speed 3 [rad/s^2] | null                |          |        |                                           |                                   |
| derivative motor speed 4 [rad/s^2] | null                |          |        |                                           |                                   |
| battery voltage [V]                | null                |          |        |                                           |                                   |

## Neurobem

- [Splash page](https://rpg.ifi.uzh.ch/NeuroBEM.html)
- [Source files and dataset](https://download.ifi.uzh.ch/rpg/NeuroBEM/)
- [Dataset readme](https://rpg.ifi.uzh.ch/neuro_bem/Readme.html)

### Note

- 96v trajectories
- 1h:15min flight time

### Measurements

> The inertial frame used in this work has a z-axis pointing upwards. Similarly, the body frame of the drone is a
> front-left-up frame, i.e. x points forwards, y to the left and the z-axis points in the direction of the propeller
> thrust.

- Vicom:
    - Captured at 400hz
    - millimeter precision
- IMU: Captured at 1khz
- Motor: Captured at 1khz

### Post-processing:

- IMU & Vicom fuse: interpolated cubic splines fitted to datapoints
- unobserved linear velocity and angular acceleration: differentiation of the fitted splines provides (not direct
  differentiation).
- time synchronization: "offset and clock skew are estimated through the correlation quality of the axis-wise angular
  rate measurement from the IMU with the spline".
- "Gyroscope measurements are used because they provide better noise characteristics than the accelerometer data".
- motor data: "smoothed with a finite-impulse-response fourth-order Butterworth low-pass filter with a cutoff frequency
  corresponding to the time-constant of the motors, identified from the step response of the motors".
-

### CSV header:

| Quantity                           | Header Abbreviation | Frame | Source      | Post-process         | Notes |
|------------------------------------|---------------------|-------|-------------|----------------------|-------|
| time [s]                           | t                   |       |             | time synchronization |       |
| position x [m]                     | pos x               |       | Vicom       |                      |       |
| position y [m]                     | pos y               |       | ...         |                      |       |
| position z [m]                     | pos z               |       | ...         |                      |       |
| quaternion qx                      | qx                  |       | Vicom       |                      |       |
| quaternion qy                      | qy                  |       | ...         |                      |       |
| quaternion qz                      | qz                  |       | ...         |                      |       |
| quaternion qw                      | qw                  |       | ...         |                      |       |
| linear velocity x [m/s]            | vel x               | body  | Vicom       | filtered             |       |
| linear velocity y [m/s]            | vel y               | ...   | ...         | ...                  |       |
| linear velocity z [m/s]            | vel z               | ...   | ...         | ...                  |       |
| linear acceleration x [m/s^2]      | acc x               | body  | IMU & Vicon | Fuse and filtered    |       |
| linear acceleration y [m/s^2]      | acc y               | ...   | ...         | ...                  |       |
| linear acceleration z [m/s^2]      | acc z               | ...   | ...         | ...                  |       |
| angular velocity x [rad/s]         | ang vel x           | body  | IMU & Vicon | Fuse and filtered    |       |
| angular velocity y [rad/s]         | ang vel y           | ...   | ...         | ...                  |       |
| angular velocity z [rad/s]         | ang vel z           | ...   | ...         | ...                  |       |
| angular acceleration x [rad/s^2]   | ang acc x           | body  | IMU & Vicon | Fuse and filtered    |       |
| angular acceleration y [rad/s^2]   | ang acc y           | ...   | ...         | ...                  |       |
| angular acceleration z [rad/s^2]   | ang acc z           | ...   | ...         | ...                  |       |
| motor speed back right [rad/s]     | mot 1               |       |             | smoothed             |       |
| motor speed front right [rad/s]    | mot 2               |       |             | ...                  |       |
| motor speed back left  [rad/s]     | mot 3               |       |             | ...                  |       |
| motor speed front left [rad/s]     | mot 4               |       |             | ...                  |       |
| derivative motor speed 1 [rad/s^2] | dmot 1              |       |             | smoothed             |       |
| derivative motor speed 2 [rad/s^2] | dmot 2              |       |             | ...                  |       |
| derivative motor speed 3 [rad/s^2] | dmot 3              |       |             | ...                  |       |
| derivative motor speed 4 [rad/s^2] | dmot 4              |       |             | ...                  |       |
| battery voltage [V]                | vbat                |       |             |                      |       |

## Models

|                             | PI-TCN                                 | NeuroBem               | Long-horizon                                         | Ours |
|-----------------------------|----------------------------------------|------------------------|------------------------------------------------------|------|
| History length (Best)       | 20                                     | 20                     | 20                                                   |      |
| Horizon length (Best)       | 1                                      | 1                      | 10 (AR unroll)                                       |      |
| Delta time                  | 10.0 ms (equaly spaced)                | 2.5 ms (equaly spaced) |                                                      |      |
|                             |                                        |                        |                                                      |      |
| Model input size            | 14 * History length                    | 10 * History length    | 14 * History length                                  |      |
| Model output size           | 6                                      | 7                      | 6 + 4                                                |      |
| Model input dims            | linear velocity                        | linear velocity        | linear velocity                                      |      |
|                             | angular velocity                       | angular velocity       | angular velocity                                     |      |
|                             | quaternion                             | motor speed            | quaternion                                           |      |
|                             | motor speed                            |                        | motor speed                                          |      |
|                             |                                        |                        |                                                      |      |
| Model output dims           | linear acceleration                    | residual force         | linear velocity                                      |      |
|                             | angular acceleration                   | residual torque        | angular velocity                                     |      |
|                             |                                        |                        | quaternion                                           |      |
|                             |                                        |                        |                                                      |      |
| Model type                  | Encoder: TCN + PIN                     | Encoder: TCN           | Encoder: TCN,GRU, LSTM or MLP                        |      |
|                             | Decoder: MLP                           | Decoder: MLP           | Decoder: MLP                                         |      |
| Model architecture and size | Encoder: 4L x 16 hid                   |                        | Encoder(GRU,LSTM) : 3L x 512 hid                     |      |
|                             |                                        |                        | Encoder(TCN): 3L x [512, 256, 256] hid               |      |
|                             |                                        |                        | Encoder(MLP): 3L x [1024, 512, 512] hid              |      |
|                             | Decoder: 3L x [64,32,32] hid           |                        | Decoder: 3L x [512,256,256] hid                      |      |
|                             | Target TCN 25k param and MLP 30k param |                        | Target param bound for real-time: 5.2 millions param |      |
|                             | Single prediction head                 | Double prediction head | Split network (velocity/attitude)                    |      |
|                             |                                        |                        |                                                      |      |
|                             |                                        |                        |                                                      |      |
| Optimization                | ReLu                                   | LeakyReLu              | LeakyReLu                                            |      |
|                             | batch norm                             |                        | No input normalization                               |      |
|                             | dropout 10%                            |                        |                                                      |      |
|                             | Adam                                   | Adam                   | AdamW                                                |      |
|                             | ?                                      |                        | 50K / (400000/512) = 64 epoch                        |      |
|                             | constant lr 1e-4                       |                        | Warmup lr 1e-4 constant 5k iter                      |      |
|                             |                                        |                        | and cosine annealing lr scheduler                    |      |
|                             | batch size 1024                        |                        | batch size 512                                       |      |
|                             |                                        |                        | Weight decay 1e-4                                    |      |
|                             |                                        |                        | Multi-step loss                                      |      |
|                             |                                        |                        | Scaled motor data by 0.001                           |      |
|                             |                                        |                        |                                                      |      |
| Compounded predictions      | No                                     | No                     | Yes                                                  | Yes  |
| Real test rollout           | Yes                                    | Yes                    | No                                                   | No   |
| Offline test rollout        | Yes                                    | Yes                    | No                                                   | Yes  |
|                             |                                        |                        |                                                      |      |
| Comments                    |                                        |                        |                                                      |      |
|                             |                                        |                        |                                                      |      |

#### Acronym

- PIN: Physic-Inspired-Neural-network
- TCN: Temporal-Convolutional-Neural-network
- MLP: Multi-Layer-Perceptron
