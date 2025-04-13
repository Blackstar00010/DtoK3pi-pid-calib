#include <TH1F.h>
#include <RooRealVar.h>
#include <RooGaussian.h>
#include <RooPolynomial.h>
#include <RooExponential.h>
#include <RooCrystalBall.h>
#include <RooDstD0BG.h>
#include <RooAddPdf.h>
#include "../utils/strops.cpp"

/*
* Create a Gaussian model for the signal
* @param x: RooRealVar pointer to the independent variable
* @param dataMean: Mean value of the data
* @param dataSigma: Sigma value of the data
* @param dataMin: Minimum value of the data
* @param dataMax: Maximum value of the data
* @param name: Name of the signal model. Default is "signal"
* @return Tuple of RooAbsPdf pointer to the signal model, and 2 RooRealVar pointers to the parameters
*/
std::tuple<RooAbsPdf*, RooRealVar*, RooRealVar*> gaussBuilder(
    RooRealVar* x, Double_t dataMean, Double_t dataSigma, Double_t dataMin, Double_t dataMax, 
    const std::string& name = "signal"
) {
    RooRealVar* mean = new RooRealVar(("mean" + name).c_str(), "Mean of Gaussian", dataMean, dataMin, dataMax);
    RooRealVar* sigma = new RooRealVar(("sigma" + name).c_str(), "Sigma of Gaussian", dataSigma / 2, 0.01, dataMax - dataMin);
    RooGaussian* gauss = new RooGaussian(name.c_str(), "Gaussian", *x, *mean, *sigma);
    return {gauss, mean, sigma};
}

/*
* Create a Crystal Ball model for the signal. See https://root.cern.ch/doc/master/classRooCrystalBall.html for more details.
* @param x: RooRealVar pointer to the independent variable
* @param dataMean: Mean value of the data
* @param dataSigma: Sigma value of the data
* @param dataMin: Minimum value of the data
* @param dataMax: Maximum value of the data
* @param name: Name of the signal model. Default is "signal"
* @return Tuple of RooAbsPdf pointer to the signal model, and 7 RooRealVar pointers to the parameters
*/
std::tuple<RooAbsPdf*, RooRealVar*, RooRealVar*, RooRealVar*, RooRealVar*, RooRealVar*, RooRealVar*, RooRealVar*> cbBuilder(
    RooRealVar* x, Double_t dataMean, Double_t dataSigma, Double_t dataMin, Double_t dataMax, 
    const std::string& name = "signal"
) {
    RooRealVar* mean = new RooRealVar(("mean" + name).c_str(), "Mean of Crystal Ball", dataMean, dataMin, dataMax);
    RooRealVar* sigmaL = new RooRealVar(("sigmaL" + name).c_str(), "SigmaL of CB", dataSigma / 2, 0.0001, dataMax - dataMin);
    RooRealVar* sigmaR = new RooRealVar(("sigmaR" + name).c_str(), "SigmaR of CB", dataSigma / 2, 0.0001, dataMax - dataMin);
    RooRealVar* alphaL = new RooRealVar(("alphaL" + name).c_str(), "AlphaL of CB", 1.0, 0.0001, 10.0);
    RooRealVar* alphaR = new RooRealVar(("alphaR" + name).c_str(), "AlphaR of CB", 1.0, 0.0001, 10.0);
    RooRealVar* nL = new RooRealVar(("nL" + name).c_str(), "nL of CB", 1.0, 0.0001, 200.0);
    RooRealVar* nR = new RooRealVar(("nR" + name).c_str(), "nR of CB", 1.0, 0.0001, 200.0);
    RooAbsPdf* crystalBall = new RooCrystalBall(name.c_str(), "Crystal Ball", *x, *mean, *sigmaL, *alphaL, *nL, *alphaR, *nR);
    return {crystalBall, mean, sigmaL, sigmaR, alphaL, alphaR, nL, nR};
}

/*
* Create a linear model for the background
* @param x: RooRealVar pointer to the independent variable
* @param nBins: Number of bins for the histogram
* @param count: Number of entries in the histogram
* @param name: Name of the background model. Default is "background"
* @return Tuple of RooAbsPdf pointer to the background model, and 2 RooRealVar pointers to the parameters
*/
std::tuple<RooAbsPdf*, RooRealVar*, RooRealVar*> linBuilder(
    RooRealVar* x, int nBins, int count, 
    const std::string& name = "background"
) {
    RooRealVar* slope = new RooRealVar(("slope" + name).c_str(), "Slope of Linear", 0, -10, 10);
    RooRealVar* intercept = new RooRealVar(("intercept" + name).c_str(), "Intercept of Linear", count / nBins, 0, count);
    RooPolynomial* linear = new RooPolynomial(name.c_str(), "Linear for X", *x, RooArgList(*slope, *intercept));
    return {linear, slope, intercept};
}

/*
* Create an exponential model for the background
* @param x: RooRealVar pointer to the independent variable
* @param name: Name of the background model. Default is "background"
* @return Tuple of RooAbsPdf pointer to the background model, and 1 RooRealVar pointer to the parameter
*/
std::tuple<RooAbsPdf*, RooRealVar*> expBuilder(
    RooRealVar* x, const std::string& name = "background"
) {
    RooRealVar* tau = new RooRealVar(("tau" + name).c_str(), "Decay constant of Exponential", -0.1, -10, 0.0);
    RooExponential* exponential = new RooExponential(name.c_str(), "Exponential", *x, *tau);
    return {exponential, tau};
}

/*
* Create a DstD0BG model for the background. See https://root.cern.ch/doc/master/classRooDstD0BG.html for more details.
* @param x: RooRealVar pointer to the independent variable
* @param dataMin: Minimum value of the data
* @param dataMax: Maximum value of the data
* @param name: Name of the background model. Default is "background"
* @return Tuple of RooAbsPdf pointer to the background model, and 4 RooRealVar pointers to the parameters
*/
std::tuple<RooAbsPdf*, RooRealVar*, RooRealVar*, RooRealVar*, RooRealVar*> dstd0bgBuilder(
    RooRealVar* x, Double_t dataMin, Double_t dataMax, 
    const std::string& name = "background"
) {
    RooRealVar* dm0 = new RooRealVar(("dm0" + name).c_str(), "Minimum delta M", dataMin + 0.4, dataMin, dataMax);
    RooRealVar* a = new RooRealVar(("a" + name).c_str(), "A of DstD0BG", 0.5, 0.0, 10.0);
    RooRealVar* b = new RooRealVar(("b" + name).c_str(), "B of DstD0BG", 0.5, 0.0, 10.0);
    RooRealVar* c = new RooRealVar(("c" + name).c_str(), "C of DstD0BG", 0.5, 0.0, 10.0);
    RooDstD0BG* dstd0bg = new RooDstD0BG(name.c_str(), "DstD0BG", *x, *dm0, *a, *b, *c);
    return {dstd0bg, dm0, a, b, c};
}


/*
* Generate a model for the signal or background
* @param modelType: Type of the model to generate
* @param hist: Histogram of the data to set the initial parameters, not to bound the model's usage
* @param x: RooRealVar pointer to the independent variable
* @param name: Name of the model. Default is "model"
* @return RooAbsPdf pointer to the model
*/
RooAbsPdf* GenerateModel(
    const std::string& modelType, TH1F* hist, RooRealVar& x, const std::string& name = "model"
) {
    Double_t min = hist->GetXaxis()->GetXmin();
    Double_t max = hist->GetXaxis()->GetXmax();
    int nBins = hist->GetNbinsX();
    int nEntries = hist->GetEntries();
    Double_t mean = hist->GetMean();
    Double_t rms = hist->GetRMS();

    std::cout << "Generating " << name << " model using " << modelType << "..." << std::endl;
    RooAbsPdf* signal = nullptr;
    if (strops::strin("gaussian", modelType)) {
        auto [gaussptr, meanptr, sigmaptr] = gaussBuilder(&x, mean, rms, min, max, name);
        signal = gaussptr;
    } else if (strops::strin("crystalballcb", modelType)) {
        auto [cbptr, meanptr, sLp, sRp, aLp, aRp, nLp, nRp] = cbBuilder(&x, mean, rms, min, max, name);
        signal = cbptr;
    } else if (strops::strin("dst0bg dstd0bkg deltambg db deltambkg", modelType)) {
        auto [dstptr, dm0ptr, aptr, bptr, cptr] = dstd0bgBuilder(&x, min, max, name);
        signal = dstptr;
    } else if (strops::strin("exponential", modelType)) {
        auto [expptr, tauptr] = expBuilder(&x, name);
        signal = expptr;
    } else if (strops::strin("linear", modelType)) {
        auto [linptr, slopeptr, interceptptr] = linBuilder(&x, nBins, nEntries, name);
        signal = linptr;
    } else {
        std::cerr << "Error: Unknown  model '" << modelType << "'" << std::endl;
        return nullptr;
    }
    if (!signal) {
        std::cerr << "Error: Model not created!" << std::endl;
        return nullptr;
    }
    std::cout << "Model " << modelType << " generated!" << std::endl;
    return signal;
}

// TH1D version as TH1D and TH1F are not automatically convertible; idk why; shitty
RooAbsPdf* GenerateModel(
    const std::string& modelType, TH1D* hist, RooRealVar& x, const std::string& name = "model"
) {
    Double_t min = hist->GetXaxis()->GetXmin();
    Double_t max = hist->GetXaxis()->GetXmax();
    int nBins = hist->GetNbinsX();
    int nEntries = hist->GetEntries();
    Double_t mean = hist->GetMean();
    Double_t rms = hist->GetRMS();

    std::cout << "Generating " << name << " model using " << modelType << "..." << std::endl;
    RooAbsPdf* signal = nullptr;
    if (strops::strin("gaussian", modelType)) {
        auto [gaussptr, meanptr, sigmaptr] = gaussBuilder(&x, mean, rms, min, max, name);
        signal = gaussptr;
    } else if (strops::strin("crystalballcb", modelType)) {
        auto [cbptr, meanptr, sLp, sRp, aLp, aRp, nLp, nRp] = cbBuilder(&x, mean, rms, min, max, name);
        signal = cbptr;
    } else if (strops::strin("dstd0bg dstd0bkg deltambg db deltambkg", modelType)) {
        auto [dstptr, dm0ptr, aptr, bptr, cptr] = dstd0bgBuilder(&x, min, max, name);
        signal = dstptr;
    } else if (strops::strin("exponential", modelType)) {
        auto [expptr, tauptr] = expBuilder(&x, name);
        signal = expptr;
    } else if (strops::strin("linear", modelType)) {
        auto [linptr, slopeptr, interceptptr] = linBuilder(&x, nBins, nEntries, name);
        signal = linptr;
    } else {
        std::cerr << "Error: Unknown  model '" << modelType << "'" << std::endl;
        return nullptr;
    }
    if (!signal) {
        std::cerr << "Error: Model not created!" << std::endl;
        return nullptr;
    }
    std::cout << "Model " << modelType << " generated!" << std::endl;
    return signal;
}
// TH1D version as TH1D and TH1F are not automatically convertible; idk why; shitty
RooAbsPdf* GenerateModel(
    const std::string& modelType, TH1D* hist, RooRealVar* x, const std::string& name = "model"
) {
    Double_t min = hist->GetXaxis()->GetXmin();
    Double_t max = hist->GetXaxis()->GetXmax();
    int nBins = hist->GetNbinsX();
    int nEntries = hist->GetEntries();
    Double_t mean = hist->GetMean();
    Double_t rms = hist->GetRMS();

    std::cout << "Generating " << name << " model using " << modelType << "..." << std::endl;
    RooAbsPdf* signal = nullptr;
    if (strops::strin("gaussian", modelType)) {
        auto [gaussptr, meanptr, sigmaptr] = gaussBuilder(x, mean, rms, min, max, name);
        signal = gaussptr;
    } else if (strops::strin("crystalballcb", modelType)) {
        auto [cbptr, meanptr, sLp, sRp, aLp, aRp, nLp, nRp] = cbBuilder(x, mean, rms, min, max, name);
        signal = cbptr;
    } else if (strops::strin("dstd0bg dstd0bkg deltambg db deltambkg", modelType)) {
        auto [dstptr, dm0ptr, aptr, bptr, cptr] = dstd0bgBuilder(x, min, max, name);
        signal = dstptr;
    } else if (strops::strin("exponential", modelType)) {
        auto [expptr, tauptr] = expBuilder(x, name);
        signal = expptr;
    } else if (strops::strin("linear", modelType)) {
        auto [linptr, slopeptr, interceptptr] = linBuilder(x, nBins, nEntries, name);
        signal = linptr;
    } else {
        std::cerr << "Error: Unknown  model '" << modelType << "'" << std::endl;
        return nullptr;
    }
    if (!signal) {
        std::cerr << "Error: Model not created!" << std::endl;
        return nullptr;
    }
    std::cout << "Model " << modelType << " generated!" << std::endl;
    return signal;
}

void models() {
    std::string thisFileName = "models.cpp";
    std::cout << thisFileName << " is not meant to run directly." << std::endl;
}