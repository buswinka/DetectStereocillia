import src.train
import src.evaluate
from src.dataloader import MaskData
import src.transforms as t
import src.utils

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import skimage.io

from torch.utils.data import DataLoader
import torchvision

import warnings
import argparse
import sys
import PySimpleGUI as sg

matplotlib.use('TkAgg')
warnings.filterwarnings("ignore")


def main(train_data, epochs, train_maskrcnn, eval):

    if epochs is not None:
        epochs = int(epochs)

    if train_maskrcnn:
        transforms = torchvision.transforms.Compose([
            t.random_h_flip(),
            t.random_v_flip(),
            t.random_affine(),
            t.adjust_brightness(),
            t.adjust_contrast(),
            t.correct_boxes()
        ])
        if train_data is None:
            raise OSError('Argument "--train_data" must be passed.')
        data = MaskData(train_data, transforms=transforms)
        data = DataLoader(data, batch_size=None, shuffle=False, num_workers=4)

        src.train.train_mask_rcnn(data, epochs=epochs)

    if eval:
        src.evaluate.evaluate(eval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", help='Number of epochs to train.')
    parser.add_argument("--train_mask_rcnn", help='Do we train mask_rcnn from scratch?', action="store_true")
    parser.add_argument("-t", "--train_all", help='Do we train a all models from scratch?', action="store_true")
    parser.add_argument("-d", "--train_data", help='Location of training data')
    parser.add_argument("-e", '--eval', help='Path to image file to analyze.')
    args = parser.parse_args()

    print(args)
    # If any arg is passed do this thing
    if not len(sys.argv) > 1:
        main(args.train_data, args.epochs, args.train_mask_rcnn, args.eval)

    sg.theme('Dark Blue 3')  # please make your windows colorful

    # layout = [[sg.Text('Your typed chars appear here:'), sg.Text(size=(12, 1), key='-OUTPUT-')],
    #           [sg.Input(key='-IN-')],
    #           [sg.Image(r'/media/DataStorage/Dropbox (Partners HealthCare)/DetectStereocillia/data/train/16k01-1.png',
    #                     size=(500,500)),],
    #           [sg.Button('Show'), sg.Button('Exit')]]
    #
    # window = sg.Window('Window Title', layout)
    #
    # while True:  # Event Loop
    #     event, values = window.read()
    #     print(event, values)
    #     if event == sg.WIN_CLOSED or event == 'Exit':
    #         break
    #     if event == 'Show':
    #         # change the "output" element to be the value of "input" element
    #         window['-OUTPUT-'].update(values['-IN-'])
    #
    # window.close()

    path = r'/media/DataStorage/Dropbox (Partners HealthCare)/DetectStereocillia/data/train/16k01-1.png'
    im = skimage.io.imread(path)
    print(im.shape)

    fig = plt.figure(frameon=False, edgecolor='Black')
    plt.axis('off')
    plt.tight_layout(w_pad=0, h_pad=0)
    plt.imshow(im)


    # ------------------------------- END OF YOUR MATPLOTLIB CODE -------------------------------

    # ------------------------------- Beginning of Matplotlib helper code -----------------------

    def draw_figure(canvas, figure):
        figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
        figure_canvas_agg.draw()
        figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
        return figure_canvas_agg


    # ------------------------------- Beginning of GUI CODE -------------------------------

    # define the window layout
    layout = [[sg.Text('Plot test')],
              [sg.Canvas(key='-CANVAS-')],
              [sg.Button('Ok')]]

    # create the form and show it without the plot
    window = sg.Window('Demo Application - Embedding Matplotlib In PySimpleGUI', layout, finalize=True,
                       element_justification='center', font='Helvetica 18')

    # add the plot to the window
    fig_canvas_agg = draw_figure(window['-CANVAS-'].TKCanvas, fig)

    event, values = window.read()

    window.close()