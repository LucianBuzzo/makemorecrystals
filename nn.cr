require "num"
require "./multinomial"

Num::Rand.set_seed(2147483647)

def one_hot(input, num_classes)
  out = Tensor(Float64, CPU(Float64)).zeros([input.size, num_classes])
  input.each_with_index do |v, i|
    out[i, v] = 1
  end
  out
end

# Define an array to hold the lines
lines = [] of String

# Open the file
File.open("names.txt") do |file|
  # Read the file, line by line
  file.each_line do |line|
    # Push each line to the array
    lines << line.chomp # chomp removes the newline character
  end
end

chars = ('a'..'z').to_a

stoi = Hash(Char | String, Int32).new
chars.each_with_index do |letter, i|
  stoi[letter] = i + 1
end
stoi["."] = 0

itos = Hash(Int32, Char | String).new
stoi.each do |k, v|
  itos[v] = k
end

xs = [] of Int32
ys = [] of Int32

lines.each do |word|
  chs = [".", *word.chars, "."]
  chs.each_cons_pair do |ch1, ch2|
    bigram = {ch1.to_s, ch2.to_s}
    ix1 = stoi[ch1]
    ix2 = stoi[ch2]
    xs << ix1
    ys << ix2
  end
end

xs = xs.to_tensor
ys = ys.to_tensor

puts "number of samples: #{xs.size}"

xenc = one_hot(xs, 27)

w = Tensor(Float64, CPU(Float64)).normal([27, 27])

probs = Tensor(Float64, CPU(Float64)).zeros([27, 27])

50.times do |epoch|
  # Forward pass
  logits = xenc.matmul(w)
  # softmax
  counts = logits.exp
  probs = counts / counts.sum(axis: 1, dims: true)

  loss = [] of Float64
  probs.shape[0].times { |k|
    loss << probs[k, ys[k].value].first
  }

  loss = 0 - loss.to_tensor.log.mean + 0.01 * (w**2).mean

  puts "loss #{loss}"

  # Calculate the softmax gradients
  dprobs = probs.dup

  dprobs.shape[0].times do |i|
    dprobs[i, ys[i].value] -= 1.0
  end

  dprobs /= ys.size

  # Now calculate the gradients for weight w
  dw = xenc.transpose.matmul(dprobs)

  # Set a learning rate
  learning_rate = 50

  # Update the weights using gradient descent
  w -= dw * learning_rate
end

10.times do
  outs = [] of (String | Char)

  ix = 0
  while true
    p = probs[ix]
    ix = Multinomial.sample(p.to_a, 1).first
    outs << itos[ix]

    break if ix == 0
  end

  puts outs.join
end
