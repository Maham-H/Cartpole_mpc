import numpy as np
import matplotlib.axes as Axes

class cartpole():

    def __init__(self):

        """ Adopted from code in:
                    @Misc{PythonLinearNonLinearControl,
                    author = {Shunichi Sekiguchi},
                    title = {PythonLinearNonlinearControl},
                    note = "\url{https://github.com/Shunichi09/PythonLinearNonlinearControl}",
                    }
        """

        self.config = {
            "states": 4,
            "inputs": 1,
            "dt": 0.01,
            "steps": 10,
            "Mc": 1, #Mass of the cart
            "mp": 0.2, #Mass of the pendulum
            "l": 1, # length of the pendulum
            "g": 9.81,
            "cart" : (1,0.75),
        }

        super(cartpole,self).__init__(self.config)

    def init_state(self,init_q=None):
        """

        :param init_q:
        :return:
        """
        self.step_count=0
        theta = np.random.randn(1)

        self.curr_q = np.array([0., 0., theta[0], 0.])

        if init_q is not None:
            self.curr_q = init_q

        # defining goal state
        self.goal_state = np.array([0., 0., -np.pi, 0.])

        # clearing history
        self.history_q =[]
        self.historu_gq = []

        return self.curr_q, {"goal_state" : self.goal_state}

    def step(self,u):
        """

        :param u: input
        :return:
        """

        # state x
        dq1 = self.curr_q[1] # initial state of the cartpole

        # state xdot
        dq2 = (u[0] + self.config["mp"]*np.sin(self.curr_q[2])*
                (self.config["l"]*(self.curr_q[3]**2)
                 + self.config["g"]*np.cos(self.curr_q[2]))
                 )/(self.config["Mc"] + self.config["mp"] *
                    (np.sin(self.curr_q[2])**2))

        # state theta
        dq3 = self.curr_q[3]

        # state thetadot
        dq4 = (-u[0] * np.cos(self.curr_q[2])
               - self.config["mp"] * self.config["l"] * (self.curr_q[3]**2)
               * np.cos(self.curr_q[2]) * np.sin(self.curr_q[2])
               - (self.config["Mc"] + self.config["mp"]) * self.config["g"]
               * np.sin(self.curr_q[2]))\
              /(self.config["l"] * (self.config["mc"] + self.config["mp"]
                                    * (np.sin(self.curr_q[2])**2)))

        next_x = self.curr_q + np.array([dq1,dq2,dq3,dq4])*self.config["dt"]


        #costs?
        cost =0.
        cost+= 0.1*np.sum(u**2)
        cost+= 6.*self.curr_q[0]**2 + 12.*(np.cos(self.curr_q[2]+1)**2)+ 0.1*self.curr_q[1]**2 +0.1*self.curr_q[3]**2

        #history
        self.history_q.append(next_x.flatten())
        self.historu_gq.append(self.goal_state.flatten())

        # update
        self.curr_x = next_x.flatten().copy()

        #update costs
        self.step_count +=1



        return next_x.flatten(),cost, self.step_count > self.config["steps"], {"goal_state":self.goal_state}

    def plot_c(self,to_plot, i=None, history_q=None, history_gq=None):
        """

        :param plot:
        :param i:
        :param history_q:
        :param history_gq:
        :return:
        """
        if isinstance(to_plot,Axes):
            imgs ={}
            imgs["cart"] = to_plot.plot([],[],c="g")[0]
            imgs["pend"] = to_plot.plot([], [], c="r", linewidth=5)[0]
            imgs["center"] = to_plot.plot([], [], marker="o",c="g", markersize=10)[0]

            to_plot.plot(np.linespace(-1.,1.,num=50),np.zeros(50),c="k",linestyle="dashed")

            #axis
            to_plot.set_xlim([-1., 1.])
            to_plot.set_ylim([-1., 1.])

            return imgs

        #
        cart_x,cart_y,pend_x, pend_y = self.coord_cartpole(history_q[i])

        to_plot["cart"].set_data(cart_x,cart_y)
        to_plot["pend"].set_data(pend_x,pend_y)
        to_plot["center"].set_data(history_q[i][0],0.)

    def coord_cartpole(self,curr_q):
        """

        :param curr_q:
        :return:
        """
        #cart
        cart_x,cart_y = self.square(curr_q[0], 0., self.config["cart"], 0.)

        #pend
        pend_x = np.array([curr_q[0], curr_q[0]+self.config["l"]*np.cos(curr_q[2]-np.pi/2)])

        pend_y = np.array([0., self.config["l"]*np.sin(curr_q[2]-np.pi/2)])

        return cart_x,cart_y, pend_x, pend_y


    #In below instances: copied code as is from

    def rotate_pos(self, pos, angle):
        """ Transformation the coordinate in the angle

        Args:
            pos (numpy.ndarray): local state, shape(data_size, 2)
            angle (float): rotate angle, in radians
        Returns:
            rotated_pos (numpy.ndarray): shape(data_size, 2)
        """
        rot_mat = np.array([[np.cos(angle), -np.sin(angle)],
                            [np.sin(angle), np.cos(angle)]])

        return np.dot(pos, rot_mat.T)

    def square(self,center_x, center_y, shape, angle):
        """

        :param center_y:
        :param shape:
        :param angle:
        :return:
        """

        square_xy = np.array([[shape[0], shape[1]],
                              [-shape[0], shape[1]],
                              [-shape[0], -shape[1]],
                              [shape[0], -shape[1]],
                              [shape[0], shape[1]]])
        # translate position to world
        # rotation
        trans_points = self.rotate_pos(square_xy, angle)
        # translation
        trans_points += np.array([center_x, center_y])

        return trans_points[:, 0], trans_points[:, 1]

