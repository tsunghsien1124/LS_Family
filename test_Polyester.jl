using BenchmarkTools
using Polyester
using Random
Random.seed!(1124)

i_size = 1
a_1_size, a_2_size, a_3_size = 10, 10, 10
a_1_grid, a_2_grid, a_3_grid = rand(a_1_size), rand(a_2_size), rand(a_3_size) 
loop_a = collect(Iterators.product(1:a_1_size, 1:a_2_size, 1:a_3_size))

b_1_size, b_2_size, b_3_size = 10, 10, 10
b_1_grid, b_2_grid, b_3_grid = rand(b_1_size), rand(b_2_size), rand(b_3_size) 

res1 = zeros(a_1_size, a_2_size, a_3_size);
res2 = zeros(a_1_size, a_2_size, a_3_size);
Eres = rand(i_size, a_1_size, a_2_size, a_3_size, b_1_size, b_2_size, b_3_size);

function test1!(res, Eres, i_size, a_1_size, a_2_size, a_3_size, a_1_grid, a_2_grid, a_3_grid, b_1_size, b_2_size, b_3_size, b_1_grid, b_2_grid, b_3_grid)
    for i in 1:i_size
        @batch for a_1_i in 1:a_1_size, a_2_i in 1:a_2_size, a_3_i in 1:a_3_size
            res[a_1_i, a_2_i, a_3_i] = 0.0
            a_1, a_2, a_3 = a_1_grid[a_1_i], a_2_grid[a_2_i], a_3_grid[a_3_i]
            for b_1_i in 1:b_1_size, b_2_i in 1:b_2_size, b_3_i in 1:b_3_size
                b_1, b_2, b_3 = b_1_grid[b_1_i], b_2_grid[b_2_i], b_3_grid[b_3_i]
                res[a_1_i, a_2_i, a_3_i] += i * (a_1 + a_2 + a_3) * b_1 * b_2 * b_3 * Eres[i, a_1_i, a_2_i, a_3_i, b_1_i, b_2_i, b_3_i]
            end
        end
    end
end

function test2!(res, Eres, i_size, loop_a, a_1_grid, a_2_grid, a_3_grid, b_1_size, b_2_size, b_3_size, b_1_grid, b_2_grid, b_3_grid)
    for i in 1:i_size
        @batch for (a_1_i, a_2_i, a_3_i) in loop_a
            res[a_1_i, a_2_i, a_3_i] = 0.0
            a_1, a_2, a_3 = a_1_grid[a_1_i], a_2_grid[a_2_i], a_3_grid[a_3_i]
            for b_1_i in 1:b_1_size, b_2_i in 1:b_2_size, b_3_i in 1:b_3_size
                b_1, b_2, b_3 = b_1_grid[b_1_i], b_2_grid[b_2_i], b_3_grid[b_3_i]
                res[a_1_i, a_2_i, a_3_i] += i * (a_1 + a_2 + a_3) * b_1 * b_2 * b_3 * Eres[i, a_1_i, a_2_i, a_3_i, b_1_i, b_2_i, b_3_i]
            end
        end
    end
end

# make the results are the same
test1!(res1, Eres, i_size, a_1_size, a_2_size, a_3_size, a_1_grid, a_2_grid, a_3_grid, b_1_size, b_2_size, b_3_size, b_1_grid, b_2_grid, b_3_grid)
test2!(res2, Eres, i_size, loop_a, a_1_grid, a_2_grid, a_3_grid, b_1_size, b_2_size, b_3_size, b_1_grid, b_2_grid, b_3_grid)
@assert all(res1 .== res2)

# benchmark the results
@btime test1!($res1, $Eres, $i_size, $a_1_size, $a_2_size, $a_3_size, $a_1_grid, $a_2_grid, $a_3_grid, $b_1_size, $b_2_size, $b_3_size, $b_1_grid, $b_2_grid, $b_3_grid)
@btime test2!($res2, $Eres, $i_size, $loop_a, $a_1_grid, $a_2_grid, $a_3_grid, $b_1_size, $b_2_size, $b_3_size, $b_1_grid, $b_2_grid, $b_3_grid)