import paramiko
import time

def create_ssh_client():
    server = '192.168.254.29'
    port = 22
    user = 'robot'
    password = 'maker'
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(server, port, username=user, password=password)
    return client


def send_command_via_shell(shell, command):
    shell.send(command + '\n')  # Send the command and newline to execute it
    
    output = read_shell_output(shell)
        
    while "True" not in output:
        print(output)
        output += read_shell_output(shell)
        if "True" in output:
            print("We are done moving")
            break
    


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

    send_command_via_shell(shell, "brush -50")
    print('Robot is ready for commands.')
    return shell
    # Example of how to send commands interactively


