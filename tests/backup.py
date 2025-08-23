    def calculate_all_rewards_penalties(self, theta, alpha):

        prev_grid = self.prev_grid
        current_grid = self.grid

        rp_matrix = np.zeros_like(self.reliability_weights)



        # Control for the case of the first step where there is no previous grid
        if isinstance(prev_grid, np.ndarray):
            prev_counts = {s: np.sum(prev_grid == s) for s in self.STATES}
        else:
            return rp_matrix 
        
        current_counts = {s: np.sum(current_grid == s) for s in self.STATES}
               
        # Get current and previous maxima
        t_max_prev = max(prev_counts, key=prev_counts.get)
        t_max_current = max(current_counts, key=current_counts.get)
        
        # Determine the Ct parameter
        # ----------------------------------
        if t_max_current != t_max_prev:
            ct = 1.0
        else:
            prev_dominant_count = prev_counts[t_max_prev]
            current_dominant_count = current_counts[t_max_current]

            delta_t = current_dominant_count - prev_dominant_count
            ct = 1.0 + (delta_t / current_dominant_count)
        # ----------------------------------

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                # For cell (i,j), we evaluate the advice its neighbors gave in prev_grid

                # Find the neighbors and their total number
                neighbourhood = self.get_valid_neighbors_with_indices(i, j)                
                num_neighbors = len(neighbourhood)
                
                # Set up a dictionary to count the states in the neighbourhood
                local_prev_counts = {s: 0 for s in self.STATES}
           
                # Fill the local_prev_counts dictionary
                for neighbor in neighbourhood:
                    ni, nj = neighbor['coord']
                    weight_idx = neighbor['weight_idx']

                    neighbor_prev_state = prev_grid[ni, nj]

                    # Update the relative state count in local_prev_counts
                    local_prev_counts[prev_grid[ni, nj]] += 1                        
                
                n_x_t_max = local_prev_counts[t_max_prev]
                numerator = num_neighbors - n_x_t_max
                # Determine the reward-penalty for the specific neighbor from the perspective of the initial cell
                for neighbor in neighbourhood:

                    n_x_neighbor_prev_state = local_prev_counts[neighbor_prev_state]                    
                    denominator = num_neighbors * n_x_neighbor_prev_state
                    fraction = numerator/ denominator

                    # Determine the Chi variable
                    # ------------------------------------
                    if neighbor_prev_state == t_max_current:
                        chi = 1
                    elif neighbor_prev_state == self.HOLD and neighbor_prev_state != t_max_current:
                        chi = -alpha
                    else:
                        chi = -(1 - alpha)    
                    # ------------------------------------   

                    rp = chi * ct * theta * fraction
                    rp_matrix[i, j, weight_idx] = rp                     

        return rp_matrix