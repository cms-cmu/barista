class plot_config:

    def __init__(self):

        self.hists               = None
        self.axisLabels          = None
        self.cutList             = None
        self.plotConfig          = None
        self.outputFolder        = None
        self.fileLabels          = None
        self.plotModifiers       = None
        self.combine_input_files = False
        self.hist_key            = "hists"

    def set_hist_key(self, hist_key):
        self.hist_key = hist_key
        self.axisLabels  = self.axisLabelsDict[hist_key]
        self.cutList     = self.cutListDict[hist_key]
