import requests
from bs4 import BeautifulSoup
import re
import numpy as np
import pandas as pd
import random
import time
from dateutil.parser import parse
import sys
sys.path.append('../')
import os
from argparse import ArgumentParser, SUPPRESS
import numpy as np
from openvino.inference_engine import IECore
from tf_model.modelBitcoin import *
from tf_model.params import *

# Check latest Bitcoin data on Yahoo Finance 
# IF current date is not in CSV file, will be added
def checkBitcoinPrice():
    url = f"https://finance.yahoo.com/quote/BTC-USD/history?p=BTC-USD"
    soup = BeautifulSoup(requests.get(url).text, "html.parser")

    class_ = "BdT Bdc($seperatorColor) Ta(end) Fz(s) Whs(nw)"
    newData = soup.find("tr", class_=class_)

    dates = []
    for td in newData:
        dates.append(td.get_text())

    current_data = dates[0]
    dt = parse(current_data)

    dates[1] = dates[1].replace(',', '') # Open
    dates[2] = dates[2].replace(',', '') # High
    dates[3] = dates[3].replace(',', '') # Low
    dates[4] = dates[4].replace(',', '') # Close
    dates[5] = dates[5].replace(',', '') # Adj Close
    dates[6] = dates[6].replace(',', '') # Volume

    newdf = pd.DataFrame([[dt.date(), dates[1], dates[2], dates[3], dates[4], dates[5], dates[6]]],
                         columns=["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"])

    df = pd.read_csv('../data/BTC-USD.csv')

    flagAddDate = True
    for i in df['Date'].values:
        prs = parse(i)
        if prs.date() == dt.date():
            flagAddDate = False

    if(flagAddDate):
        print("[INFO] Latest bitcoin data was added to CVS - {0} : Price {1} and Volumes {2}".format(dt.date(), dates[5], dates[6]))
        df = df.append(newdf, ignore_index=True)
        df.to_csv('../data/BTC-USD.csv', index=False)
    else:
        print("CSV file is up to date !")
    
    return dates


def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group("Options")
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument("-m", "--model", help="Required. Path to an .xml file with a trained model.",
                      required=True, type=str)
    args.add_argument("-l", "--cpu_extension",
                      help="Optional. Required for CPU custom layers. "
                           "Absolute path to a shared library with the kernels implementations.",
                      type=str, default=None)
    args.add_argument("-d", "--device",
                      help="Optional. ONLY CPU is acceptable ",
                      default="CPU", type=str)

    return parser


def main():    
    args = build_argparser().parse_args()

    print("Loading Inference Engine")
    ie = IECore()

    model_xml = args.model
    model_bin = os.path.splitext(model_xml)[0] + ".bin"
    print("Loading network files:\n\t{}\n\t{}".format(model_xml, model_bin))
    net = ie.read_network(model=model_xml, weights=model_bin)

    print("Device info:")
    versions = ie.get_versions(args.device)
    print("{}{}".format(" " * 8, args.device))
    print("{}MKLDNNPlugin version ......... {}.{}".format(" " * 8, versions[args.device].major,
                                                          versions[args.device].minor))
    print("{}Build ........... {}".format(" " * 8, versions[args.device].build_number))

    if args.cpu_extension and "CPU" in args.device:
        ie.add_extension(args.cpu_extension, "CPU")
        print("CPU extension loaded: {}".format(args.cpu_extension))

    if "CPU" in args.device:
        supported_layers = ie.query_network(net, "CPU")
        not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
        if len(not_supported_layers) != 0:
            print("Following layers are not supported by the plugin for specified device {}:\n {}".
                      format(args.device, ', '.join(not_supported_layers)))

    print("inputs number: " + str(len(net.input_info.keys())))

    for input_key in net.input_info:
        print("input shape: " + str(net.input_info[input_key].input_data.shape))
        print("input key: " + input_key)

    out_blob = next(iter(net.outputs))

    input_name = input_key
    net.input_info[input_key].precision = 'FP32'
    print("Precision set to :", net.input_info[input_key].precision)

    print("\n===== Check latest data from Yahoo Finance ofr Bitcoin ====\n")
    ld = checkBitcoinPrice()

    print("\n===== Read CSV data ====\n")
    data = readData(wsize=WSIZE, lookup=LOOKUP, tsize=TSIZE, dfcolumns=DFCOLUMNS, shuffle=False)

    # Prepare data for prediction
    forPrediction = data["forPrediction"][-WSIZE:]
    scalerCol = data["minmaxScaler"]
    forPrediction = forPrediction.reshape((forPrediction.shape[1], forPrediction.shape[0]))
    forPrediction = np.expand_dims(forPrediction, axis=0)

    print("\n===== Load model to device ====\n")
    exec_net = ie.load_network(network=net, device_name=args.device)
    dataInput = {}
    dataInput[input_name] = forPrediction
    print("Input Shape:", forPrediction.shape)
    res = exec_net.infer(inputs=dataInput)

    # Processing output blobs
    res = res[out_blob]

    predictedPrice = scalerCol["Adj Close"].inverse_transform(res)[0][0]
    
    print("\n===== Input and Predicted Values ====\n")
    print("[Input] Values : Open {} High {} Low {} Close {} Adj Close {} Volume {}".format(ld[1], ld[2], ld[3], ld[4], ld[5], ld[6]))
    print(f"[Output] Predicted price for tomorrow is {predictedPrice:.2f} $")

    print("\n===== Create and Load Model ====\n")
    m = createModel(dfcolumns=DFCOLUMNS, wsize=WSIZE)
    m = tf.keras.models.load_model('../tf_model/saved_model_TF')

    print("\n\n===== Plot data & Model evaluation ====\n\n")     
    mse, mae = m.evaluate(data["X_test"], data["y_test"], verbose=1)
    maeVal = data["minmaxScaler"]["Adj Close"].inverse_transform([[mae]])[0][0]
    print("Mean Absolute Error:", maeVal)
    print("Accuracy :", getAccuracy(m, data, LOOKUP))
   
    plotPrediction(m, data, predictedPrice)

if __name__ == '__main__':
    sys.exit(main() or 0)
