import keyboard

from ImageRec import *
from MovementController import next_command_from_state, extract_ball_positions, find_shortest_path, nearest_neighbor_path
from RobotConnection import create_ssh_client, create_shell, send_command_via_shell
import time

coolDownTime = 10

isConnected = True
def main():
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

    try:
        while True:
            current_time = time.time()

            if keyboard.is_pressed('q'):
                print("Termination requested. Exiting...")
                send_command_via_shell(shell, "brush 0")
                break

            

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
                

                if command and command.lower() == 'exit':
                    print("catching")
                    break
                if command and command.lower() != "":
                    print(command)  # send_command_via_shell(shell, command)
                    if ssh_client is not None and shell is not None:
                        send_command_via_shell(shell, command)
                # Instead of sleep, use a loop to frequently check for interrupts
                sleep_duration = 2  # total sleep time in seconds
                sleep_step = 0.1  # sleep step in seconds
                for _ in range(int(sleep_duration / sleep_step)):
                    if keyboard.is_pressed('q'):
                        print("Termination requested during sleep. Exiting...")
                        send_command_via_shell(shell, "brush 0")
                        raise KeyboardInterrupt
                    time.sleep(sleep_step)
            else:
                cool -= 1
    except KeyboardInterrupt:
        print("Program interrupted by user.")
        if ssh_client is not None and shell is not None:
            send_command_via_shell(shell, "brush 0")
    finally:
        if ssh_client is not None and shell is not None:
            shell.close()
            ssh_client.close()

if __name__ == '__main__':
    main()
