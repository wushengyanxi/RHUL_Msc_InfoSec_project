import socket

Terminal = socket.socket(family = socket.AF_INET, type = socket.SOCK_STREAM)
print(socket.SOL_SOCKET)
Terminal.connect(("127.0.0.1",8081))

while True:
    message = input(">>: ").strip()

    if not message:
        continue

    Terminal.send(message.encode("utf-8"))
    data = Terminal.recv(128)
    print(data.decode("utf-8"))

Terminal.close()