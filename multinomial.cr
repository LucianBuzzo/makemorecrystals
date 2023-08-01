module Multinomial
  # Define the multinomial class method
  def self.sample(probabilities : Array(Float64), num_samples : Int32? = nil)
    # If num_samples is provided, draw multiple samples
    if num_samples
      return Array.new(num_samples) { draw_sample(probabilities) }.to_tensor
    end

    # If num_samples is not provided, draw a single sample
    [draw_sample(probabilities)].to_tensor
  end

  private def self.draw_sample(probabilities : Array(Float64))
    # Generate a random number between 0 and 1
    rand_num = Random.new.rand

    # Calculate the cumulative probabilities
    cumulative_prob = 0.0

    # Loop through the probabilities
    probabilities.each_with_index do |prob, index|
      cumulative_prob += prob
      return index if rand_num <= cumulative_prob
    end

    # If no index has been returned, return the last one
    probabilities.size - 1
  end
end
