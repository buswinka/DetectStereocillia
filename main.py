import src.train
import src.evaluate
from src.dataloader import MaskData
import src.transforms as t
import src.utils

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as Tk
import skimage.io as io

from torch.utils.data import DataLoader
import torchvision

import warnings
import argparse
import sys
import PySimpleGUI as sg

import os.path

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

def draw_figure_w_toolbar(canvas, fig):
    if canvas.children:
        for child in canvas.winfo_children():
            child.destroy()

    figure_canvas_agg = FigureCanvasTkAgg(fig, master=canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side=Tk.RIGHT, fill=Tk.BOTH, expand=1)

    def on_key_press(event):
        canvas.TKCanvas.mpl_connect("key_press_event", on_key_press)
    return

class Toolbar(NavigationToolbar2Tk):
    # only display the buttons we need
    toolitems = [t for t in NavigationToolbar2Tk.toolitems if
                 t[0] in ('Home', 'Pan', 'Zoom')]
    # t[0] in ('Home', 'Pan', 'Zoom','Save')]

    def __init__(self, *args, **kwargs):
        super(Toolbar, self).__init__(*args, **kwargs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", help='Number of epochs to train.')
    parser.add_argument("--train_mask_rcnn", help='Do we train mask_rcnn from scratch?', action="store_true")
    parser.add_argument("-t", "--train_all", help='Do we train a all models from scratch?', action="store_true")
    parser.add_argument("-d", "--train_data", help='Location of training data')
    parser.add_argument("-e", '--eval', help='Path to image file to analyze.')
    args = parser.parse_args()

    # If any arg is passed do this thing
    if len(sys.argv) > 1:
        main(args.train_data, args.epochs, args.train_mask_rcnn, args.eval)

    else:

        # Code from: https://github.com/SuperMechaDeathChrist/Widgets/blob/master/plt_figure_w_controls.py

        eval = src.evaluate.evaluate()

        sg.theme('Dark Blue 3')  # please make your windows colorful

        layout = [
            [sg.T('Analyze Stereocilia App')],
            [sg.Text("Image: "), sg.In(size=(70, 1), enable_events=True, key="-FOLDER-"), sg.FileBrowse(key='-FILE-')],
            [sg.B('Analyze'), sg.B('Save Analysis'), sg.B('Exit')],
            [sg.T('Figure:')],
            [sg.Column(layout=[[sg.Canvas(key='fig_cv',size=(400 * 2, 400))]],background_color='#DAE0E6',pad=(0, 0))],
        ]

        window = sg.Window(title='Graph with controls', layout=layout)
        window.Finalize()
        window.Maximize()

        im = []

        while True:
            event, values = window.Read()

            if event in [None, 'Exit']:  # always,  always give a way out!
                window.Close()
                break

            elif event == '-FOLDER-':
                plt.figure(1)
                fig = plt.gcf()
                DPI = fig.get_dpi()
                im = io.imread(values['-FILE-'])
                plt.imshow(im,cmap='Greys_r')
                ax = plt.gca()
                ax.axes.xaxis.set_visible(False)
                ax.axes.yaxis.set_visible(False)
                plt.tight_layout()
                draw_figure_w_toolbar(window.FindElement('fig_cv').TKCanvas, fig)

            elif event == 'Analyze':
                try:
                    out, masks = eval(values['-FILE-'])
                    out = out.transpose((1,2,0))
                except (AttributeError, RuntimeError):
                    continue

                plt.figure(1)
                fig = plt.gcf()
                DPI = fig.get_dpi()
                plt.imshow(out)
                ax = plt.gca()
                ax.axes.xaxis.set_visible(False)
                ax.axes.yaxis.set_visible(False)
                plt.tight_layout()
                draw_figure_w_toolbar(window.FindElement('fig_cv').TKCanvas, fig)

            elif event == 'Save Analysis':
                print('SAVED YEET')
                filename = values['-FILE-']
                dir = os.path.splitext(filename)[0]
                os.mkdir(dir)


