import keyboard

from ImageRec import *
from MovementController import next_command_from_state
from RobotConnection import create_ssh_client, create_shell

coolDownTime = 10

isConnected = False
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
    while True:
        if keyboard.is_pressed('q'):
            break
        state = render()
        if cool < 0:
            cool = coolDownTime
            command = next_command_from_state(state)
            if command.lower() == 'exit':
                break
            if command.lower() != "":
                print(command)  # send_command_via_shell(shell, command)
            if ssh_client is not None and shell is not None:
                print(" ")
        else:
            cool -= 1

    if ssh_client is not None and shell is not None:
        shell.close()
        ssh_client.close()


if __name__ == '__main__':
    main()
