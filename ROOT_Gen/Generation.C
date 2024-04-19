#include <TSystem.h>

void Generation(int n, int nh, int nw)
{
    // image size (nh x nw)
    const int ntot = nh * nw;
    const TString folderName = "images";
    const TString fileOutName = TString::Format("%s/images_data_%dx%d_%d.root", folderName.Data(), nh, nw, n);

    // Check if folder exists, if not create it
    gSystem->mkdir(folderName, kTRUE);

    TFile f(fileOutName, "RECREATE");

    const int nRndmEvts = 10000; // number of events we use to fill each image
    double delta_sigma = 0.1;    // 5% difference in the sigma
    double pixelNoise = 5;

    double sX1 = 3;
    double sY1 = 3;
    double sX2 = sX1 + delta_sigma;
    double sY2 = sY1 - delta_sigma;

    TH2D h1("h1", "h1", nh, 0, 10, nw, 0, 10);
    TH2D h2("h2", "h2", nh, 0, 10, nw, 0, 10);

    TF2 f1("f1", "xygaus");
    TF2 f2("f2", "xygaus");

    TTree sgn("sig_tree", "signal_tree");
    TTree bkg("bkg_tree", "background_tree");

    std::vector<float> x1(ntot);
    std::vector<float> x2(ntot);

    // create signal and background trees with a single branch
    // an std::vector<float> of size nh x nw containing the image data

    std::vector<float> *px1 = &x1;
    std::vector<float> *px2 = &x2;

    bkg.Branch("vars", "std::vector<float>", &px1);
    sgn.Branch("vars", "std::vector<float>", &px2);

    sgn.SetDirectory(&f);
    bkg.SetDirectory(&f);

    f1.SetParameters(1, 5, sX1, 5, sY1);
    f2.SetParameters(1, 5, sX2, 5, sY2);
    gRandom->SetSeed(0);
    std::cout << "Filling ROOT tree " << std::endl;
    for (int i = 0; i < n; ++i) {
        if (i % 1000 == 0) 
            std::cout << "Generating image event ... " << i << std::endl;
        h1.Reset();
        h2.Reset();
        // generate random means in range [3,7] to be not too much on the border
        f1.SetParameter(1, gRandom->Uniform(3, 7));
        f1.SetParameter(3, gRandom->Uniform(3, 7));
        f2.SetParameter(1, gRandom->Uniform(3, 7));
        f2.SetParameter(3, gRandom->Uniform(3, 7));

        h1.FillRandom("f1", nRndmEvts);
        h2.FillRandom("f2", nRndmEvts);

        for (int k = 0; k < nh; ++k) {
            for (int l = 0; l < nw; ++l) {
                int m = k * nw + l;
                // add some noise in each bin
                x1[m] = h1.GetBinContent(k + 1, l + 1) + gRandom->Gaus(0, pixelNoise);
                x2[m] = h2.GetBinContent(k + 1, l + 1) + gRandom->Gaus(0, pixelNoise);
            }
        }
        sgn.Fill();
        bkg.Fill();
    }
    sgn.Write();
    bkg.Write();

    Info("MakeImagesTree", "Signal and background tree with images data written to the file %s", f.GetName());
    sgn.Print();
    bkg.Print();
    f.Close();
}
