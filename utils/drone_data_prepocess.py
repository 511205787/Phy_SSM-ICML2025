from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import os

# Reference: https://github.com/arplaboratory/data-driven-system-identification

######################################################################################
# example for crazyflie
model_name = "crazyflie" # "crazyflie"
output_topic = "actuator_motors_mux"
inertia_ratio = 1.832
exponents_thrust_curve = [[0, 1, 2], [2]] if model_name == "crazyflie" else [[0, 1, 2]]
dpi=800
debug=False
create_motor_delay_animation=False

if model_name == "crazyflie":
    rotor_x_displacement = 0.028
    rotor_y_displacement = 0.028
    model = {
        "gravity": 9.81,
        "mass": 0.027,
        "rotor_positions": np.array([
            [ rotor_x_displacement, -rotor_y_displacement, 0],
            [-rotor_x_displacement, -rotor_y_displacement, 0],
            [-rotor_x_displacement,  rotor_y_displacement, 0],
            [ rotor_x_displacement,  rotor_y_displacement, 0]
        ]), # eq 5. r_pi
        "rotor_thrust_directions": np.array([
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1]
        ]), # eq 5. r_{f_i}?
        "rotor_torque_directions": np.array([
            [0, 0, -1],
            [0, 0,  1],
            [0, 0, -1],
            [0, 0,  1]
        ]) # eq 5. r_{tau_i}
    }
elif model_name == "large":
    rotor_x_displacement = 0.4179/2
    rotor_y_displacement = 0.481332/2
    # model is in FLU frame
    model = {
        "gravity": 9.81,
        "mass": 2.3 + 1.05,
        "rotor_positions": np.array([
            [ rotor_x_displacement, -rotor_y_displacement, 0],
            [-rotor_x_displacement,  rotor_y_displacement, 0],
            [ rotor_x_displacement,  rotor_y_displacement, 0],
            [-rotor_x_displacement, -rotor_y_displacement, 0]
        ]),
        "rotor_thrust_directions": np.array([
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1]
        ]),
        "rotor_torque_directions": np.array([
            [0, 0, -1],
            [0, 0, -1],
            [0, 0,  1],
            [0, 0,  1]
        ])
    }
    g = np.array([0, 0, -model["gravity"]])


if model_name == "crazyflie":
    log_files = [
        "data/logs_crazyflie/log10",
        "data/logs_crazyflie/log15",
        "data/logs_crazyflie/log16",
    ]
elif model_name == "large":
    ulog_files = [
        "data/logs_large/log_63_2024-1-8-16-37-54.ulg",
        "data/logs_large/log_64_2024-1-8-16-39-44.ulg",
        "data/logs_large/log_65_2024-1-8-16-40-52.ulg",
        "data/logs_large/log_66_2024-1-8-16-42-48.ulg",
    ]
wanted_columns = [
    *[f"vehicle_acceleration_xyz[{i}]" for i in range(3)],
    *[f"vehicle_angular_velocity_xyz[{i}]" for i in range(3)],
    *[f"vehicle_angular_velocity_xyz_derivative[{i}]" for i in range(3)],
    *[f"{output_topic}_control[{i}]" for i in range(4)],
]

if model_name == "crazyflie":
    import os
    if "COLAB_GPU" in os.environ and not os.path.exists("cfusdlog.py"): # check if running in colab and if yes download cfusdlog parser automagically
        import requests
        file_url = 'https://github.com/bitcraze/crazyflie-firmware/raw/f61da11d54b6d54c7fa746688e8e9be4edd73a29/tools/usdlog/cfusdlog.py'
        local_filename = 'cfusdlog.py'
        response = requests.get(file_url, stream=True)
        with open(local_filename, 'wb') as file:
            for chunk in response.iter_content(chunk_size=128):
                file.write(chunk)
    import utils.cfusdlog as cfusdlog # use the download_cfusdlog.sh script to download cfusdlog.py
    decoded_data = [cfusdlog.decode(f) for f in log_files]
    flights = []
    for name, data in zip(log_files, decoded_data):
        flight = {}
        flight["name"] = name
        flight["convention"] = "flu"
        flight["data"] = {}
        timestamps = data["fixedFrequency"]["timestamp"] / 1000
        for key, value in data["fixedFrequency"].items():
            if key.startswith("acc"):
                if key[-1:] == "x":
                    flight["data"]["vehicle_acceleration_xyz[0]"] = {"timestamps": timestamps, "values": np.array(value) * 9.81} # cf accelerometer logs are in gs
                if key[-1:] == "y":
                    flight["data"]["vehicle_acceleration_xyz[1]"] = {"timestamps": timestamps, "values": np.array(value) * 9.81}
                if key[-1:] == "z":
                    flight["data"]["vehicle_acceleration_xyz[2]"] = {"timestamps": timestamps, "values": np.array(value) * 9.18}
            elif key.startswith("gyro"):
                if key[-1:] == "x":
                    flight["data"]["vehicle_angular_velocity_xyz[0]"] = {"timestamps": timestamps, "values": np.array(value)/360*2*np.pi} # cf gyro logs are in degrees/s
                if key[-1:] == "y":
                    flight["data"]["vehicle_angular_velocity_xyz[1]"] = {"timestamps": timestamps, "values": np.array(value)/360*2*np.pi}
                if key[-1:] == "z":
                    flight["data"]["vehicle_angular_velocity_xyz[2]"] = {"timestamps": timestamps, "values": np.array(value)/360*2*np.pi}
            elif key.startswith("motor"):
                flight["data"][f"{output_topic}_control[{int(key[-1:])-1}]"] = {"timestamps": timestamps, "values": np.array(value) / 65536}
        flights.append(flight)
elif model_name in ["large"]:
    import os, sys, subprocess
    from utils.load_ulg import load_ulg
    dfs = [load_ulg(file) for file in ulog_files]
    flights = [
        {
            "name": name,
            "convention": "frd",
            "data": {
                name: {
                    "timestamps": np.array(c.index),
                    "values": np.array(c)
                } for name, c in [(wc, df[wc].dropna()) for wc in wanted_columns]
            }
        } for name, df in zip(ulog_files, dfs)
    ]


for flight_i, flight in enumerate(flights):
    fig, axs = plt.subplots(4, 1, figsize=(10, 8), sharex=True)

    ax = 0
    for motor_i, motor in enumerate([f"{output_topic}_control[{i}]" for i in range(4)]):
        axs[ax].plot(flight["data"][motor]["timestamps"], flight["data"][motor]["values"], label=f"Motor {motor_i}")
        axs[ax].legend(loc='upper left', bbox_to_anchor=(1,1))
        axs[ax].set_ylabel('Motor RPM setpoints')
    ax += 1
    thrust_values = []
    thrust_torque_values = []
    geometric_torque_values = []
    timestamps = None
    for motor_i, motor in enumerate([f"{output_topic}_control[{i}]" for i in range(4)]):
        timestamps = flight["data"][motor]["timestamps"]
        thrust_values.append(flight["data"][motor]["values"][:, np.newaxis] * model["rotor_thrust_directions"][motor_i][np.newaxis, :]) # eq2
        thrust_torque_values.append(flight["data"][motor]["values"][:, np.newaxis] * model["rotor_torque_directions"][motor_i][np.newaxis, :]) # eq 5 second term
        geometric_torque_values.append(flight["data"][motor]["values"][:, np.newaxis] * np.cross(model["rotor_positions"][motor_i], model["rotor_thrust_directions"][motor_i])[np.newaxis, :]) # eq 5 first term
    thrust_values = sum(thrust_values)
    thrust_torque_values = sum(thrust_torque_values)
    geometric_torque_values = sum(geometric_torque_values)

    axs[ax].set_title("Linear Dynamics Excitation")
    axs[ax].plot(timestamps, thrust_values[:, 2], label="z")
    axs[ax].legend(loc='upper left', bbox_to_anchor=(1,1))
    axs[ax].set_ylabel('Excitation level')
    ax += 1
    
    axs[ax].set_title("Angular Dynamics Excitation (x, y)")
    axs[ax].plot(timestamps, geometric_torque_values[:, 0], label="x")
    axs[ax].legend(loc='upper left', bbox_to_anchor=(1,1))
    axs[ax].set_ylabel('Geometry Torque (x)')
    axs[ax].plot(timestamps, geometric_torque_values[:, 1], label="y")
    axs[ax].legend(loc='upper left', bbox_to_anchor=(1,1))
    axs[ax].set_ylabel('Excitation level')
    ax += 1
    
    axs[ax].set_title("Angular Dynamics Excitation (z)")
    axs[ax].plot(timestamps, thrust_torque_values[:, 2], label="z")
    axs[ax].legend(loc='upper left', bbox_to_anchor=(1,1))
    axs[ax].set_ylabel('Excitation level')
    ax += 1

    axs[-1].set_xlabel('Time [s]')
    for ax in axs[:-1]:
        ax.xaxis.set_tick_params(labelbottom=True)
    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    plt.savefig(f"figures/{model_name}_flight{flight_i}.pdf")
    plt.savefig(f"figures/{model_name}_flight{flight_i}.jpg", dpi=dpi)
    fig.suptitle(f"Flight {flight_i}: {flight['name']}")
    plt.tight_layout(rect=[0, 0, 1, 0.99])

#################################################
# select time slice
##################################################
if model_name == "crazyflie":
    timeframes_thrust = [
        {
            "flight": 0,
            "start": 35,
            "end": 65
        }
    ]
    timeframes_inertia_roll_pitch = [
        {
            "flight": 1,
            "start": 55,
            "end": 90
        }
    ]
    timeframes_inertia_yaw = [
        {
            "flight": 2,
            "start": 32,
            "end": 38
        },
        {
            "flight": 2,
            "start": 45,
            "end": 49
        }
    ]
elif model_name == "large":
    timeframes_thrust = [
        {
            "flight": 0,
            "start": 10,
            "end": 45
        }
    ]
    timeframes_inertia_roll_pitch = [
        {
            "flight": 1,
            "start": 10,
            "end": 17
        },
        {
            "flight": 1,
            "start": 19,
            "end": 35
        },
        {
            "flight": 2,
            "start": 10,
            "end": 45
        }
    ]
    timeframes_inertia_yaw = [
        {
            "flight": 3,
            "start": 15,
            "end": 30
        }
    ]

def extract_timeframes(flights, timeframes):
    output_flights = []
    fragment_counter = {}
    for timeframe in timeframes:
        flight = deepcopy(flights[timeframe["flight"]])
        fragment_id = 0 if flight["name"] not in fragment_counter else fragment_counter[flight["name"]]
        fragment_counter[flight["name"]] = fragment_id + 1
        flight["name"] = flight["name"] + f".{fragment_id}"
        start, end = timeframe["start"], timeframe["end"]
        data = flight["data"]
        for series in data:
            mask = (data[series]["timestamps"] > start) & (data[series]["timestamps"] < end)
            data[series]["timestamps"] = data[series]["timestamps"][mask]
            data[series]["values"] = data[series]["values"][mask]
        output_flights.append(flight)
    return output_flights

flights_thrust = extract_timeframes(flights, timeframes_thrust)
flights_inertia_roll_pitch = extract_timeframes(flights, timeframes_inertia_roll_pitch)
flights_inertia_yaw = extract_timeframes(flights, timeframes_inertia_yaw)

##################################################
# detect gap and interpolate data
##################################################
def slice_gaps_and_interpolate(flights):
    flights_output = []
    for flight in flights:
        lowest_frequency = None
        lowest_frequency_name = None
        highest_frequency = None
        highest_frequency_name = None
        for name, data in flight["data"].items():
            diff = np.diff(data["timestamps"])
            frequency = 1/np.median(diff)
            if lowest_frequency is None or frequency < lowest_frequency:
                lowest_frequency = frequency
                lowest_frequency_name = name
            if highest_frequency is None or frequency > highest_frequency:
                highest_frequency = frequency
                highest_frequency_name = name

        interval_threshold = 3 * 1/lowest_frequency
        print(f"Lowest frequency: {lowest_frequency} for {lowest_frequency_name}")
        print(f"Highest frequency: {highest_frequency} for {highest_frequency_name}")

        earliest_timestamp_all = max([data["timestamps"][0] for name, data in flight["data"].items()])
        latest_timestamp_all = min([data["timestamps"][-1] for name, data in flight["data"].items()])
        print(f"Earliest timestamp_all: {earliest_timestamp_all}")
        print(f"Latest timestamp_all: {latest_timestamp_all}")
        master_timestamps_full = flight["data"][highest_frequency_name]["timestamps"]
        master_timestamps = master_timestamps_full[(master_timestamps_full > earliest_timestamp_all) & (master_timestamps_full < latest_timestamp_all)]
        earliest_timestamp = master_timestamps[0]
        latest_timestamp = master_timestamps[-1]
        print(f"Cutting {(100 * (1 - len(master_timestamps) / len(master_timestamps_full))):.2f}% of the data to synchronize the timestamp start and end")

        total_time = latest_timestamp - earliest_timestamp

        gaps = []
        for name, data in flight["data"].items():
            current_timestamps_full = data["timestamps"]
            current_timestamps = current_timestamps_full[(current_timestamps_full > earliest_timestamp) & (current_timestamps_full < latest_timestamp)]
            current_timestamps_augmented = np.concatenate([[earliest_timestamp], current_timestamps, [latest_timestamp]])
            diff = np.diff(current_timestamps_augmented)
            current_gaps = np.where(diff > interval_threshold)[0]
            for gap in current_gaps:
                gap_start = data["timestamps"][gap]
                gap_end = data["timestamps"][gap+1]
                gaps.append((gap_start, gap_end))
        gaps_sorted = sorted(gaps, key=lambda x: x[0])


        current_gap_start = None
        current_gap_end = None
        combined_gaps = []

        for i, (gap_start, gap_end) in enumerate(gaps_sorted):
            if current_gap_start is None:
                current_gap_start = gap_start
            
            if current_gap_end is None:
                current_gap_end = gap_end
            
            if gap_end > current_gap_end:
                current_gap_end = gap_end
            
            if i < len(gaps_sorted) - 1:
                next_gap_start, next_gap_end = gaps_sorted[i+1]
                if next_gap_start - current_gap_end > interval_threshold:
                    print(f"Gap: {current_gap_start} - {current_gap_end}")
                    combined_gaps.append((current_gap_start, current_gap_end))
                    current_gap_start = None
                    current_gap_end = None
            else:
                print(f"Gap start {gap_start} - {gap_end}")
                print(f"Final Gap: {current_gap_start} - {current_gap_end}")
                combined_gaps.append((current_gap_start, current_gap_end))
        print(f"Number of gaps: {len(combined_gaps)}")

        total_gap_time = sum([gap_end - gap_start for gap_start, gap_end in combined_gaps])
        assert total_gap_time < 0.1 * total_time, f"Total gap time: {total_gap_time:.2f}s"

        subflights = []
        current_segment_start_timestamp = earliest_timestamp
        for gap_start, gap_end in [*combined_gaps, (latest_timestamp, latest_timestamp)]:
            segment_time = gap_start - current_segment_start_timestamp
            if segment_time > 0.01 * total_time:
                current_segment_timestamps = master_timestamps[(master_timestamps > current_segment_start_timestamp) & (master_timestamps < gap_start)]
                sub_flight = {
                    name: {
                        "timestamps": current_segment_timestamps,
                        "values": np.interp(current_segment_timestamps, data["timestamps"], data["values"])
                    } for name, data in flight["data"].items()
                }
                subflights.append(sub_flight)
            else:
                print(f"Skipping segment of length {segment_time:.2f}s")
            current_segment_start_timestamp = gap_end
        print(f"Number of subflights: {len(subflights)}")


        for subflight in subflights:
            plt.figure()
            plt.title(flight["name"])
            for i, key in enumerate([f"actuator_motors_mux_control[{i}]" for i in range(4)]):
                ts = subflight[key]
                plt.plot(ts["timestamps"], ts["values"], label=f"motor {i}")
            plt.legend()
        
        for subflight_i, subflight in enumerate(subflights):
            flights_output.append({
                "name": flight["name"] + f"_{subflight_i}",
                "convention": flight["convention"],
                "timestamps": subflight[highest_frequency_name]["timestamps"],
                "data": subflight
            })
    return flights_output
sliced_and_interpolated_flights_thrust = slice_gaps_and_interpolate(flights_thrust)
sliced_and_interpolated_flights_inertia_roll_pitch = slice_gaps_and_interpolate(flights_inertia_roll_pitch)
sliced_and_interpolated_flights_inertia_yaw = slice_gaps_and_interpolate(flights_inertia_yaw)

##################################################
# filter motor data
##################################################
def filter_ema(timestamps, rpm_setpoints, T_m):
    rpms_filtered = []
    rpm = None
    previous_t = None
    for t, rpm_setpoint in zip(timestamps, rpm_setpoints):
        if rpm is None:
            rpm = rpm_setpoint
        else:
            delta_t = t - previous_t
            alpha = np.exp(-delta_t / T_m)
            rpm = alpha*rpm + (1-alpha) * rpm_setpoint
        rpms_filtered.append(rpm)
        previous_t = t
    return np.array(rpms_filtered)

def combine(flights, T_m, T_omega=0.05, thrust_curves=None):
    assert thrust_curves is None or len(thrust_curves) == 4
    from scipy.spatial.transform import Rotation as R

    def FRD2FLU(x):
        return np.array([x[0], -x[1], -x[2]])

    new_flights = []
    for flight in flights:
        timestamps = flight["timestamps"]
        frd = flight["convention"] == "frd"
        rpm_setpoints = np.array([
            flight["data"][f"{output_topic}_control[{i}]"]["values"] for i in range(4)
        ]).T

        acceleration_original_frame = np.array([
            flight["data"][f"vehicle_acceleration_xyz[{i}]"]["values"] for i in range(3)
        ]).T
        acceleration = np.array(list(map(FRD2FLU, acceleration_original_frame))) if frd else acceleration_original_frame

        omega_original_frame = np.array([
            flight["data"][f"vehicle_angular_velocity_xyz[{i}]"]["values"] for i in range(3)
        ]).T
        omega = np.array(list(map(FRD2FLU, omega_original_frame))) if frd else omega_original_frame

        if f"vehicle_angular_velocity_xyz_derivative[0]" in flight["data"]:
            domega_original_frame = np.array([
                flight["data"][f"vehicle_angular_velocity_xyz_derivative[{i}]"]["values"] for i in range(3)
            ]).T
            domega = np.array(list(map(FRD2FLU, domega_original_frame))) if frd else domega_original_frame
        else:
            domega = flight["domega_gradient"] = np.gradient(omega, timestamps, axis=0)

        # Filter the RPM setpoints to get actual RPMs
        flight["rpm_setpoints"] = rpm_setpoints
        flight["rpms"] = filter_ema(timestamps, rpm_setpoints, T_m)
        flight["acceleration"] = acceleration
        flight["omega"] = omega
        flight["domega"] = domega

        # Calculate thrust and torque values
        thrust_values = []
        thrust_torque_values = []
        geometric_torque_values = []
        for motor_i in range(4):
            control_input = rpm_setpoints[:, motor_i][:, np.newaxis]  # Shape (N, 1)

            # Rotor thrust direction
            rotor_thrust_direction = model["rotor_thrust_directions"][motor_i][np.newaxis, :]  # Shape (1, 3)
            thrust = control_input * rotor_thrust_direction  # Shape (N, 3)
            thrust_values.append(thrust)

            # Rotor torque direction
            rotor_torque_direction = model["rotor_torque_directions"][motor_i][np.newaxis, :]  # Shape (1, 3)
            torque_due_to_drag = control_input * rotor_torque_direction  # Shape (N, 3)
            thrust_torque_values.append(torque_due_to_drag)

            # Geometric torque
            torque_geometric = control_input * np.cross(
                model["rotor_positions"][motor_i],
                model["rotor_thrust_directions"][motor_i]
            )[np.newaxis, :]  # Shape (N, 3)
            geometric_torque_values.append(torque_geometric)

        # Sum over all motors
        flight["thrust_values"] = np.sum(thrust_values, axis=0)  # Shape (N, 3)
        flight["thrust_torque_values"] = np.sum(thrust_torque_values, axis=0)  # Shape (N, 3)
        flight["geometric_torque_values"] = np.sum(geometric_torque_values, axis=0)  # Shape (N, 3)

        flight["timestamps"] = timestamps  # Ensure timestamps are included
        new_flights.append(flight)

    # Combine data from all flights
    combined_data = {
        "rpm_setpoints": np.concatenate([flight["rpm_setpoints"] for flight in new_flights]),
        "rpms": np.concatenate([flight["rpms"] for flight in new_flights]),
        "acceleration": np.concatenate([flight["acceleration"] for flight in new_flights]),
        "omega": np.concatenate([flight["omega"] for flight in new_flights]),
        "domega": np.concatenate([flight["domega"] for flight in new_flights]),
        "thrust_values": np.concatenate([flight["thrust_values"] for flight in new_flights]),
        "thrust_torque_values": np.concatenate([flight["thrust_torque_values"] for flight in new_flights]),
        "geometric_torque_values": np.concatenate([flight["geometric_torque_values"] for flight in new_flights]),
        "timestamps": np.concatenate([flight["timestamps"] for flight in new_flights]),
    }

    return combined_data, new_flights

# ######################################################################################
# # Process Each Flight Slice Individually and Save as Pickle
# ######################################################################################
# import pickle

# def process_and_save_flights(flights, T_m, save_dir, prefix):
#     os.makedirs(save_dir, exist_ok=True)
#     for idx, flight in enumerate(flights):
#         # Process the flight
#         combined_data, _ = combine([flight], T_m)  # Process single flight
#         # Reset timestamps to start from 0
#         start_time = combined_data["timestamps"][0]
#         combined_data["timestamps"] = combined_data["timestamps"] - start_time
#         # Save combined_data as pickle
#         filename = f"{prefix}_flight_{idx}.pkl"
#         filepath = os.path.join(save_dir, filename)
#         with open(filepath, 'wb') as f:
#             pickle.dump(combined_data, f)
#         print(f"Saved {filename}")

# # Set the motor time constant
# T_m_test = 0.05

# # Process and save flights for thrust
# process_and_save_flights(
#     sliced_and_interpolated_flights_thrust,
#     T_m_test,
#     save_dir='data/drone_data',
#     prefix='log10'
# )

# # Process and save flights for inertia roll and pitch
# process_and_save_flights(
#     sliced_and_interpolated_flights_inertia_roll_pitch,
#     T_m_test,
#     save_dir='data/drone_data',
#     prefix='log15'
# )

# # Process and save flights for inertia yaw
# process_and_save_flights(
#     sliced_and_interpolated_flights_inertia_yaw,
#     T_m_test,
#     save_dir='data/drone_data',
#     prefix='log16'
# )

# print("All flights have been processed and saved as pickle files.")


T_m_test = 0.05
combined_test, _ = combine(sliced_and_interpolated_flights_thrust, T_m_test)   # log 10 data
# sliced_and_interpolated_flights_thrust -- log 10 data
# sliced_and_interpolated_flights_inertia_roll_pitch -- log 15 data
# sliced_and_interpolated_flights_inertia_yaw -- log 16 data
######################################################################################
# Plot Combined Data (visualization)
######################################################################################
import matplotlib.pyplot as plt
import os

# Create a directory to save figures
os.makedirs("figures", exist_ok=True)

# Extract timestamps from combined data
timestamps = combined_test["timestamps"]

# Define the variables you want to plot
variables_to_plot = [
    ("rpm_setpoints", "Motor RPM Setpoints", 4),
    ("rpms", "Filtered Motor RPMs", 4),
    ("acceleration", "Acceleration", 3),
    ("omega", "Angular Velocity", 3),
    ("domega", "Angular Acceleration", 3),
    ("thrust_values", "Thrust Values", 3),
    ("geometric_torque_values", "Geometric Torque Values", 3),
    ("thrust_torque_values", "Thrust Torque Values", 3),
]

for var_name, var_label, num_components in variables_to_plot:
    fig, axs = plt.subplots(num_components, 1, figsize=(12, 3 * num_components), sharex=True)
    data = combined_test[var_name]
    for i in range(num_components):
        ax = axs[i] if num_components > 1 else axs
        ax.plot(timestamps, data[:, i], label=f"{var_label} [{i}]")
        ax.set_ylabel(f"{var_label} [{i}]")
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax.grid(True)
    axs[-1].set_xlabel('Time [s]')
    plt.tight_layout()
    plt.savefig(f"figures/{model_name}_{var_name}_plot.png", dpi=dpi)
    plt.close()

print("Plots have been generated and saved in the 'figures' directory.")

######################################################################################
# Process Each Flight Slice Individually and Save as Pickle
######################################################################################
import pickle

def process_and_save_flights(flights, T_m, save_dir, prefix):
    os.makedirs(save_dir, exist_ok=True)
    for idx, flight in enumerate(flights):
        # Process the flight
        combined_data, _ = combine([flight], T_m)  # Process single flight
        # Reset timestamps to start from 0
        start_time = combined_data["timestamps"][0]
        combined_data["timestamps"] = combined_data["timestamps"] - start_time
        # Save combined_data as pickle
        filename = f"{prefix}_flight_{idx}.pkl"
        filepath = os.path.join(save_dir, filename)
        with open(filepath, 'wb') as f:
            pickle.dump(combined_data, f)
        print(f"Saved {filename}")

# Set the motor time constant
T_m_test = 0.05

# Process and save flights for thrust
process_and_save_flights(
    sliced_and_interpolated_flights_thrust,
    T_m_test,
    save_dir='data/drone_data',
    prefix='log10'
)

# Process and save flights for inertia roll and pitch
process_and_save_flights(
    sliced_and_interpolated_flights_inertia_roll_pitch,
    T_m_test,
    save_dir='data/drone_data',
    prefix='log15'
)

# Process and save flights for inertia yaw
process_and_save_flights(
    sliced_and_interpolated_flights_inertia_yaw,
    T_m_test,
    save_dir='data/drone_data',
    prefix='log16'
)

print("All flights have been processed and saved as pickle files.")

print(1)