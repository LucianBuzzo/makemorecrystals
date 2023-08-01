require "num"
require "./multinomial"

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

N = Tensor(Int32, CPU(Int32)).zeros([27, 27])

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

lines.each do |word|
  chs = [".", *word.chars, "."]
  chs.each_cons_pair do |ch1, ch2|
    bigram = {ch1.to_s, ch2.to_s}
    ix1 = stoi[ch1]
    ix2 = stoi[ch2]
    N[ix1, ix2] += 1
  end
end

Num::Rand.set_seed(2147483647)

# Normalize the counts to get probabilities
P_ = N + 1
puts P_.shape
P = P_ / P_.sum(axis: 1, dims: true)

5.times do
  outs = [] of (String | Char)

  ix = 0
  while true
    p = P[ix]
    ix = Multinomial.sample(p.to_a, 1).first
    outs << itos[ix]

    break if ix == 0
  end

  puts outs.join
end

log_likelihood = 0.0
n = 0
lines[0...3].each do |word|
  chs = [".", *word.chars, "."]
  chs.each_cons_pair do |ch1, ch2|
    bigram = {ch1.to_s, ch2.to_s}
    ix1 = stoi[ch1]
    ix2 = stoi[ch2]
    prob = P[ix1, ix2]
    logprob = Num.log(prob)
    log_likelihood += logprob
    n += 1
    puts "#{bigram} #{prob} #{logprob}"
  end
end

puts "log likelihood: #{log_likelihood}"
nll = 0 - log_likelihood
puts "negative log likelihood: #{nll}"
puts "loss: #{nll / n}"
