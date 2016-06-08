class Backpropagation
  def self.trainingCycle! (net, trainingIn, trainingOut, epochs, eta, alpha)
    @@ihDeltas = []
    @@hoDeltas = []
    net.inputs.length.times do |i|
      @@ihDeltas[i] = []
      net.hidden.length.times do |h|
        @@ihDeltas[i][h] = 0;
      end
    end
    net.hidden.length.times do |h|
      @@hoDeltas[h] = []
      net.output.length.times do |o|
        @@hoDeltas[h][o] = 0;
      end
    end

    acc = [] #accuracy
    0.upto(epochs) do |e|

      trainingIn.length.times do |i|
        output = net.propagate(trainingIn[i])
        puts output
        self.train!(net, output, trainingOut[i], eta, alpha)

      end

    end
  end

  def self.train! (net, output, idealOut, eta, alpha)
    oGradients = []
    hGradients = []
    idealOut.length.times do |i|
      oGradients[i] = (idealOut[i] - output[i]) * net.activationDirivative(i, 2)
    end
    net.hidden.length.times do |i|
      sum = 0
      idealOut.length.times do |j|
        sum += oGradients[j] * net.hoWeights[i][j]
      end
      hGradients[i] = net.activationDirivative(i, 1) * sum
    end

    net.inputs.length.times do |i|
      net.hidden.length.times do |h|
        @@ihDeltas[i][h] = eta * hGradients[h] * net.inputs[i] + @@ihDeltas[i][h] * alpha
        net.ihWeights[i][h] += @@ihDeltas[i][h]
      end
    end

    net.hidden.length.times do |h|
      net.output.length.times do |o|
        @@hoDeltas[h][o] = eta * hGradients[h] * net.hidden[h] + @@hoDeltas[h][o] * alpha
        net.hoWeights[h][o] += @@hoDeltas[h][o]
      end
    end

  end

end