require 'Backpropagation'

class Network

  attr_reader :inputs, :hidden, :output, :ihWeights, :hoWeights
  attr_writer :inputs, :hidden, :output, :ihWeights, :hoWeights
  def initialize (inputs, hidden, output)
    @inputs = []
    @output = []
    @oSums = [] #sums before activation function for backpropagation
    @hidden = []
    @hSums = [];
    @ihWeights = [] #inputs-hidden
    @hoWeights = [] #hidden-output

    inputs.times do |i|
      @inputs[i] = 0;
      @ihWeights[i] ||= []
      hidden.times do |h|
        @hidden[h] = 0;
        @ihWeights[i][h] = Random.rand

        unless @hoWeights[h] then
          @hoWeights[h] ||= []
          output.times do |o|
            @output[o] ||= 0
            @hoWeights[h][o] = Random.rand
          end
        end
      end
    end

  end

  def propagate(inputs)
    @inputs.length.times do |i|
      @inputs[i] = inputs[i]
    end
    @hidden.length.times do |h|
      @hSums[h] = 0
      inputs.length.times do |i|
        @hSums[h] += inputs[i] * @ihWeights[i][h]
      end
      @hidden[h] = activation(@hSums[h])
    end

    @output.length.times do |o|
      @oSums[o] = 0
      hidden.length.times do |h|
        @oSums[o] += @hidden[h] * @hoWeights[h][o]
      end
      @output[o] = activation(@oSums[o]);
    end

    return @output
  end

  def activation(n) #activation function
    return Math.tanh(n)
  end

  def activationDirivative(i, layer)
    if(layer == 2) then
      n = @output[i]
    else
      n = @hidden[i]
    end

    return 1 - n * n
  end

end

net = Network.new(1,5,1)

trainingIn = []
100.times do |i|
  trainingIn[i] = [Random.rand]
end
trainingOut = []
trainingIn.length().times do |i|
  trainingOut[i] = [Math.sin(trainingIn[i][0])]
end
Backpropagation.trainingCycle!(net,trainingIn,trainingOut,500,0.001, 0.001)
puts "Final results: "
dist = 0#sum of errors
trainingIn.length.times do |i|
  dist += (trainingOut[i][0] - net.propagate(trainingIn[i])[0]).abs
  puts "#{trainingIn[i]}, #{trainingOut[i]}, #{net.propagate(trainingIn[i])}"
end
puts "Average error: #{dist / trainingIn.length}"
