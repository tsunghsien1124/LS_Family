function BM_function(
    action_size::Integer
    )
    """
    compute the sorted indices of binary monotonicity algorithm

    BM_index[:,1] saves the middle point
    BM_index[:,2] saves the indicator of lower bound
    BM_index[:,3] saves the indicator of upper bound
    """

    # initialize auxiliary matrix
    auxiliary_matrix = []
    push!(auxiliary_matrix, [1, action_size])

    # initialize the matrix storing binary search indices
    BM_index = zeros(Int, 2, 3)
    BM_index[1,:] = [1 1 1]
    BM_index[2,:] = [action_size 1 action_size]

    # set up criterion and iteration number
    k = Inf
    iter = 1

    while k > 1

        # initializa the number of rows, i.e., k
        if iter == 1
            k = 1
        end

        # step 2 on page 35 in Gordon and Qiu (2017, WP)
        while (auxiliary_matrix[end][1]+1) < auxiliary_matrix[end][2]
            m = convert(Int, floor((auxiliary_matrix[end][1]+auxiliary_matrix[end][2])/2))

            if findall(BM_index[:,1] .== m) == []
                BM_index = cat(BM_index, [m auxiliary_matrix[end][1] auxiliary_matrix[end][2]]; dims = 1)
            end

            push!(auxiliary_matrix, [auxiliary_matrix[end][1], m])
            k += 1
        end

        # step 3 on page 35 in Gordon and Qiu (2017, WP)
        if k == 1
            break
        end
        while auxiliary_matrix[end][2] == auxiliary_matrix[end-1][2]
            pop!(auxiliary_matrix)
            k -= 1
            if k == 1
                break
            end
        end
        if k == 1
            break
        end

        # step 4 on page 35 in Gordon and Qiu (2017, WP)
        auxiliary_matrix[end][1] = auxiliary_matrix[end][2]
        auxiliary_matrix[end][2] = auxiliary_matrix[end-1][2]

        # update iteration number
        iter += 1
    end

    # return results
    return BM_index
end

function HM_algorithm(
    lb::Integer,            # lower bound of choices
    ub::Integer,            # upper bound of choices
    Π::Function             # return function
    )
    """
    implement Heer and Maussner's (2005) algorithm of binary concavity
    """

    while true
        # points of considered
        n = ub - lb + 1

        if n < 1
            error("n < 1 in HM algorithm")
        end

        # step 1 on page 536 in Gordon and Qiu (2018, QE)
        if n == 1
            return lb, Π(lb)
        else
            flag_lb = 0
            flag_ub = 0
        end

        # step 2 on page 536 in Gordon and Qiu (2018, QE)
        if n == 2
            if flag_lb == 0
                Π_lb = Π(lb)
            end
            if flag_ub == 0
                Π_ub = Π(ub)
            end
            if Π_lb > Π_ub
                return lb, Π_lb
            else
                return ub, Π_ub
            end
        end

        # step 3 on page 536 in Gordon and Qiu (2018, QE)
        if n == 3
            if max(flag_lb, flag_ub) == 0
                Π_lb = Π(lb)
                flag_lb = 1
            end
            m = convert(Int, (lb+ub)/2)
            Π_m = Π(m)
            if flag_lb == 1
                if Π_lb > Π_m
                    return  lb, Π_lb
                end
                lb, Π_lb, flag_lb = m, Π_m, 1
            else # flag_ub == 1
                if Π_ub > Π_m
                    return  ub, Π_ub
                end
                ub, Π_ub, flag_ub = m, Π_m, 1
            end
        end

        # step 4 on page 536 in Gordon and Qiu (2018, QE)
        if n >= 4
            m = convert(Int, floor((lb+ub)/2))
            Π_m = Π(m)
            Π_m1 = Π(m+1)
            if Π_m < Π_m1
                lb, Π_lb, flag_lb = (m+1), Π_m1, 1
            else
                ub, Π_ub, flag_ub = m, Π_m, 1
            end
        end
    end
end
