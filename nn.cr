require "num"
require "./multinomial"

Num::Rand.set_seed(2147483647)
ctx = Num::Grad::Context(Tensor(Float64, CPU(Float64))).new

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

lines[0...1].each do |word|
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

def one_hot(input, num_classes)
  out = Tensor(Float64, CPU(Float64)).zeros([input.size, num_classes])
  input.each_with_index do |v, i|
    out[i, v] = 1
  end
  out
end

xenc = ctx.variable(one_hot(xs, 27))

w = ctx.variable(Tensor(Float64, CPU(Float64)).normal([27, 27]))
puts w

net = Num::NN::Network.new(ctx)

logits = xenc.matmul(w)
# softmax
counts = logits.exp
puts counts
probs = counts / ctx.variable(counts.value.sum(axis: 1, dims: true))

puts probs[0]
puts ys

puts Tensor.range(5)
loss = [] of Float64
5.times { |k|
  loss << probs[k, ys.to_a[k]].value.first
}

loss = 0 - loss.to_tensor.log.mean

puts loss
