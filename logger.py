import datetime


class Logger:

    def __init__(self):
        pass

    def log(self, message):
        with open("log.txt", "a") as f:
            msg = "[" + datetime.datetime.now().strftime("%Y-%m-%d %H:%m:%S") + "] " + message + '\n'
            f.write(msg)
            print(msg)


if __name__ == '__main__':
    logger = Logger()
    logger.log("sup")