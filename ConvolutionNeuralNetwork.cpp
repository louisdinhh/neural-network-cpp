// ANSI - ISO standard for C++
#include <vector>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <sstream>
#include <cassert>

using namespace std;

// class to read training data from file
class TrainingData
{
    public:
        TrainingData(const string filename);
        bool isEof(void) { return n_trainingDataFile.eof();}
        void getTopology(vector<unsigned> &topology);
    
    // Returns the number of input values read from the file:
    unsigned getNextInputs(vector<double> &inputVals);
    unsigned getTargetOutputs(vector<double> &targetOutputVals);

    private:
        ifstream n_trainingDataFile;
};

void TrainingData::getTopology(vector<unsigned> &topology)
{
    string line;
    string label;

    getline(n_trainingDataFile, line);
    stringstream ss(line);
    ss >> label;
    if (this ->isEof() || label.compare("topology") != 0){
        abort();
    }

    while (!ss.eof()) {
        unsigned n;
        ss >> n;
        topology.push_back(n);
    }
    return;
}

TrainingData::TrainingData(const string filename)
{
    n_trainingDataFile.open(filename.c_str());
}

unsigned TrainingData::getNextInputs(vector<double> &inputVals)
{
    inputVals.clear();

    string line;
    getline(n_trainingDataFile, line);
    stringstream ss(line);

    string label;
    ss >> label;
    if (label.compare("in:") == 0){
        double oneValue;
        while (ss >> oneValue){
            inputVals.push_back(oneValue);
        }
    }

    return inputVals.size();
}

unsigned TrainingData::getTargetOutputs(vector<double> &targetOutputVals)
{
    targetOutputVals.clear();

    string line;
    getline(n_trainingDataFile, line);
    stringstream ss(line);

    string label;
    ss >> label;
    if (label.compare("out:") == 0){
        double oneValue;
        while (ss >> oneValue){
            targetOutputVals.push_back(oneValue);
        }
    }
    return targetOutputVals.size();
}

struct Connection
{
    double weight;
    double deltaWeight;
};

class Neuron;
typedef vector<Neuron> Layer;

// ***************** class Neuron ********************
class Neuron
{
    public:
        Neuron( unsigned numOutputs, unsigned myIndex);
        void setOutputVal(double val) { n_outputVal = val;}
        double getOutputVal(void) const { return n_outputVal;}
        void feedForward(const Layer &prevLayer);
        void calcOutputGradients(double targetVal);
        void calcHiddenGradients(const Layer &nextLayer);
        void updateInputWeights(Layer &prevLayer);

    private:
        static double eta; // overall net training rate
        static double alpha; // multiplier of last weight change (momentum)
        static double transferFunction(double x);
        static double transferFunctionDerivative(double x);
        static double randomWeight(void) { return rand() / double(RAND_MAX); }
        double sumDOW(const Layer &nextLayer) const;
        double n_outputVal;
        vector<Connection> n_outputWeights;
        unsigned n_myIndex;
        double n_gradient;
};

double Neuron::eta = 0.15; // change training rate here
double Neuron::alpha = 0.5; // momentum multiplier of last deltaWeight



void Neuron::updateInputWeights(Layer &prevLayer)
{
    // The weight to be updated are in the Connection container, in the neurons in the preceeding layer
    for ( unsigned n = 0; n < prevLayer.size(); ++n){
        Neuron &neuron = prevLayer[n];
        double oldDeltaWeight = neuron.n_outputWeights[n_myIndex].deltaWeight;
        // Individual input, magnified by the gradient and train rate & add momentum:
        double newDeltaWeight = eta * neuron.getOutputVal() * n_gradient + alpha * oldDeltaWeight;
        neuron.n_outputWeights[n_myIndex].deltaWeight = newDeltaWeight;
        neuron.n_outputWeights[n_myIndex].deltaWeight += newDeltaWeight;
    }
}

double Neuron::sumDOW(const Layer &nextLayer) const
{
    double sum = 0.0;

    //Sum our contributions of the errors at the nodes we feed

    for (unsigned n = 0; n < nextLayer.size() - 1; ++n){
        sum += n_outputWeights[n].weight * nextLayer[n].n_gradient;
    }

    return sum;
}

void Neuron::calcHiddenGradients(const Layer &nextLayer)
{
    double dow = sumDOW(nextLayer);
    n_gradient = dow * Neuron::transferFunctionDerivative(n_outputVal);
}


void Neuron::calcOutputGradients(double targetVal)
{
    double delta = targetVal - n_outputVal;
    n_gradient = delta * Neuron::transferFunctionDerivative(n_outputVal);
}


double Neuron::transferFunction(double x)
{
    // tanh - output range [-1, 0, ... , 1, 0 ]
    return tanh(x);
}

double Neuron::transferFunctionDerivative(double x)
{
    // tanh derivative
    return 1 - x*x;
}

void Neuron::feedForward(const Layer &prevLayer)
{
    double sum = 0.0;
    //Sum the previous layer's outputs (inputs) + bias node from the previous layer
    for (unsigned n = 0; n < prevLayer.size(); ++n){
        sum += prevLayer[n].getOutputVal() *
            prevLayer[n].n_outputWeights[n_myIndex].weight;
    }

    n_outputVal = Neuron::transferFunction(sum);
}

    Neuron::Neuron( unsigned numOutputs, unsigned myIndex)
{
    for (unsigned c = 0; c < numOutputs; ++c){
        n_outputWeights.push_back(Connection());
        n_outputWeights.back().weight = randomWeight();
    }

    n_myIndex = myIndex;
}

// ***************** class Net ***********************

class Net
{
    public:
        Net(const vector<unsigned> &topology);
        void feedForward(const vector<double> &inputVals);
        void backProp(const vector<double> &targetVals);
        void getResults(vector<double> &resultVals) const;
        double getRecentAverageError(void) const { return n_recentAverageError; }

    private:
        vector<Layer> n_layers; // n_layers[layerNum][neuronNum]
        double n_error;
        double n_recentAverageError;
        double n_recentAverageSoothingFactor;
};

void Net::getResults(vector<double> &resultVals) const
{
    resultVals.clear();

    for (unsigned n = 0; n < n_layers.back().size() - 1; ++n) {
        resultVals.push_back(n_layers.back()[n].getOutputVal());
    }
}

void Net::backProp(const vector<double> &targetVals)
{
    // Calculate overall net error (RMS of output neuron errors)
    Layer &outputLayer = n_layers.back();
    n_error = 0.0;

    for (unsigned n = 0; n < outputLayer.size() - 1; ++n) {
        double delta = targetVals[n] - outputLayer[n].getOutputVal();
        n_error += delta * delta;
    }
    n_error /= outputLayer.size() - 1; // get average error squared
    n_error = sqrt(n_error); //RMS

    // Implement a recent average measurement:
    n_recentAverageError = (n_recentAverageError * n_recentAverageSoothingFactor + n_error) / (n_recentAverageSoothingFactor + 1.0);

    // Calculate output layer gradients
    for (unsigned n = 0; n < outputLayer.size() - 1; ++n) {
        outputLayer[n].calcOutputGradients(targetVals[n]);
    }

    // Calculate gradients on hidden layers
    for (unsigned layerNum = n_layers.size() - 2; layerNum > 0; --layerNum) {
        Layer &hiddenLayer = n_layers[layerNum];
        Layer &nextLayer = n_layers[layerNum + 1];

        for (unsigned n = 0; n < hiddenLayer.size(); ++n){
            hiddenLayer[n].calcHiddenGradients(nextLayer);
        }
    }

    // For all layers from outputs to first hidden layer, update connection weights
    for (unsigned layerNum = n_layers.size() - 1; layerNum > 0; --layerNum) {
        Layer &layer = n_layers[layerNum];
        Layer &prevLayer = n_layers[layerNum - 1];

        for (unsigned n = 0; n < layer.size() - 1; ++n) {
            layer[n].updateInputWeights(prevLayer);
        }
    }

}

void Net::feedForward(const vector<double> &inputVals)
{
    assert(inputVals.size() == n_layers[0].size() - 1);

    // Assign {latch} the input values into the input neurons
    for (unsigned i = 0; i < inputVals.size(); ++i) {
        n_layers[0][i].setOutputVal(inputVals[i]); 
    }

    // Forward propagate
    for (unsigned layerNum = 1; layerNum < n_layers.size(); ++layerNum){
        Layer &prevLayer = n_layers[layerNum -1];
        for (unsigned n = 0; n < n_layers[layerNum].size() - 1; ++n){
            n_layers[layerNum][n].feedForward(prevLayer);
        }
    }
}

Net::Net(const vector<unsigned> &topology)
{
    unsigned numLayers = topology.size();
    for (unsigned layerNum = 0; layerNum < numLayers; ++layerNum){
        n_layers.push_back(Layer());
        unsigned numOutputs = layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];

        //add a bias neuron to the layer
        for (unsigned neuroNum =0; neuroNum <= topology[layerNum]; ++neuroNum){
            n_layers.back().push_back(Neuron(numOutputs, neuroNum));
            cout << "Made a Neuron!" << endl;
        }

        // Force the bias node's output value to 1.0. It's the last neuron created above
        n_layers.back().back().setOutputVal(1.0);
    }
}

void showVectorVals(string label, vector<double> &v)
{
    cout << label << " ";
    for (unsigned i = 0; i < v.size(); ++i) {
        cout << v[i] << " ";
    }

    cout << endl;
}

int main()
{
    TrainingData trainData("/tmp/trainingData.txt");

    // e.g., { 3, 2, 1 }
    vector<unsigned> topology;
    trainData.getTopology(topology);

    Net myNet(topology);

    vector<double> inputVals, targetVals, resultVals;
    int trainingPass = 0;

    while (!trainData.isEof()) {
        ++trainingPass;
        cout << endl << "Pass " << trainingPass;

        // Get new input data and feed it forward:
        if (trainData.getNextInputs(inputVals) != topology[0]) {
            break;
        }
        showVectorVals(": Inputs:", inputVals);
        myNet.feedForward(inputVals);

        // Collect the net's actual output results:
        myNet.getResults(resultVals);
        showVectorVals("Outputs:", resultVals);

        // Train the net what the outputs should have been:
        trainData.getTargetOutputs(targetVals);
        showVectorVals("Targets:", targetVals);
        assert(targetVals.size() == topology.back());

        myNet.backProp(targetVals);

        // Report how well the training is working, average over recent samples:
        cout << "Net recent average error: "
                << myNet.getRecentAverageError() << endl;
    }

    cout << endl << "Done" << endl;
}
