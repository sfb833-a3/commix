def create_config(log_file):
    dictLogConfig = {}
    dictLogConfig["version"] = 1
    
    fileHandler = {
        "class":"logging.FileHandler",
        "formatter": "simpleFormatter",
        "filename": log_file
    }
    consoleHandler = {
        "class":"logging.StreamHandler",
        "formatter": "simpleFormatter",
    }
    dictLogConfig["handlers"] = {"fileHandler": fileHandler, "consoleHandler": consoleHandler}

    train_logger = {"handlers":["fileHandler", "consoleHandler"], "level":"DEBUG"}
    dictLogConfig["loggers"] = {"train":train_logger}

    formatter = {"format":"%(asctime)s - %(name)s - %(levelname)s - %(message)s", 
                'datefmt': '%Y-%m-%d %H:%M'}
    dictLogConfig["formatters"] = {"simpleFormatter": formatter}

    return dictLogConfig