import keyboard

from CopyImageRec import *
from MovementController import next_command_from_state, extract_ball_positions, find_shortest_path, nearest_neighbor_path
from RobotConnection import create_ssh_client, create_shell, send_command_via_shell
import time
import threading

coolDownTime = 10

isConnected = True


def real_program():
    ssh_client = None
    shell = None

    try:
        if isConnected:
            ssh_client = create_ssh_client()
            shell = create_shell(ssh_client)
    except Exception as e:
        print(e)

    cool = coolDownTime
    start_time = time.time()

    all_grid_calc()

    try:
        while True:
            current_time = time.time()

            if keyboard.is_pressed("q"):
                print("Termination requested. Exiting...")
                send_command_via_shell(shell, "brush 50")
                time.sleep(7)
                send_command_via_shell(shell, "brush 0")
                break
            # TODO: Add a condition to move balls to goal at 7:30
            """if tid >= 450:
                print("7:30 reached, moving robot to goal position")
                send_command_via_shell(shell, "move_to_goal_command")
                break"""

            tid = current_time - start_time

            try:
                while True:
                    state = render()
                    if state:
                        break
                    else:
                        print("State is None, retrying...")
            except Exception as e:
                send_command_via_shell(shell, "brush 0")
                print("Exception caught:", str(e))
                break

            if cool < 0:
                cool = coolDownTime
                command = None

                command = next_command_from_state(state)

                if command and command.lower() == "exit":
                    print("catching")
                    break
                if command and command.lower() != "":
                    print(command)  # send_command_via_shell(shell, command)
                    if ssh_client is not None and shell is not None:
                        send_command_via_shell(shell, command)
                        if command == "brush 80":
                            time.sleep(10)

            else:
                cool -= 1
    except KeyboardInterrupt:
        print("Program interrupted by user.")
        if ssh_client is not None and shell is not None:
            send_command_via_shell(shell, "brush 50")
            time.sleep(7)
            send_command_via_shell(shell, "brush 0")
    finally:
        if ssh_client is not None and shell is not None:
            shell.close()
            ssh_client.close()


def testing_environment():
    try:
        while True:
            state = render()
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Program interrupted by user.")


def main():

    print("Choose mode:\n1. Real Program\n2. Testing Environment")
    choice = input("Enter 1 or 2: ").strip()

    if choice == "1":
        print("Real program")

        testing_thread = threading.Thread(target=testing_environment)
        testing_thread.start()

        real_program()
    elif choice == "2":
        testing_environment()
    else:
        print("Invalid choice. Exiting...")


if __name__ == "__main__":
    main()
