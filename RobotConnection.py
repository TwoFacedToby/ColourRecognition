import paramiko
import time

def create_ssh_client():
    server = '192.168.230.29'
    port = 22
    user = 'robot'
    password = 'maker'
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(server, port, username=user, password=password)
    return client


def send_command_via_shell(shell, command):
    shell.send(command + '\n')  # Send the command and newline to execute it
    time.sleep(1)  # Allow some time for the command to be processed


def read_shell_output(shell):
    time.sleep(1)  # Allow time for data to arrive
    if shell.recv_ready():
        return shell.recv(4096).decode()  # Increase buffer size if necessary
    return ""


def create_shell(ssh_client):
    print('Creating Connection')

    print('Connection Established')

    print('Starting interactive shell')
    shell = ssh_client.invoke_shell()
    print('Interactive shell started')

    print('Starting Program')
    send_command_via_shell(shell, 'python3 /home/robot/cdio.py')

    # Wait for the robot to signal it is ready to receive orders
    output = ""
    while "Ready for orders:" not in output:
        output += read_shell_output(shell)
        if "Ready for orders:" in output:
            break

    print('Robot is ready for commands.')
    return shell
    # Example of how to send commands interactively


