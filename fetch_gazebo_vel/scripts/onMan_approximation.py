import numpy as np
# from hgrad_functions.mcbf_hgrad_res30 import *
from copy import deepcopy
from scipy.spatial.transform import Rotation as R
import threading
# from obs_gpdf_vec import GPDF


class OnMan_Approx:
    def __init__(self, target, convex, concave, w = None, radius=0, margin=0, xc=None, rho=1, obs_goal=None, it_max=None, dt = None) -> None:
        self.target = target.reshape(2,1)
        self.w = w
        self.convex = convex
        self.concave = concave
        self.margin=margin

        if convex:
            # self.convex_num = num
            self.radius = radius
            self.xc = xc
            self.rho = rho

            if obs_goal!=None:
                self.it_max = it_max
                self.step = (obs_goal - self.xc)/it_max
                self.t = 0
                self.dxc = self.step/dt
            else:
                self.step = 0
                if self.xc !=None:
                    self.dxc = np.zeros(self.xc.shape)

        if concave:
            self.gpis = GPDF()
            blob1 = self.gpis.create_obs(4., 4., 2.3)
            blob2 = self.gpis.create_obs(4., 4., 2)
            pc_coords =  np.hstack((blob1, blob2)).T
            self.gpis.update_gpdf(pc_coords)



    def receding_horizon(self,P_result, P_index, x, u, alpha, iter):
        # x dim -> dimxN
        # u dim -> dimxN
        dim = x.shape[0]
        temp_states = deepcopy(x)
        Pi = np.zeros(x.shape[1])
        for i in range(iter):
            if dim == 2:
                # print(temp_states.T.shape)
                if temp_states.shape[1]>1:
                    _,grad_x_p,_,_= self.h_grad_vector(temp_states.T)
                    grad_x_p = grad_x_p.T
                else:
                    _,grad_x_p,_,_ = self.h_grad_standard(temp_states.T)
                grad_x_p = grad_x_p.reshape(2,-1)
            else:
                grad_x_p = np.zeros((3,temp_states.shape[1]))
                grad_x_p[0] = gradp_x(temp_states.T,self.w)
                grad_x_p[1] = gradp_y(temp_states.T,self.w)
                grad_x_p[2] = gradp_theta(temp_states.T,self.w)

            E_T = self.get_basis(grad_x_p.T/np.linalg.norm(grad_x_p,axis=0).reshape(-1,1), dim-1, orth=True)
            E_T = np.moveaxis(E_T, 1, 0)
            E = np.transpose(E_T, axes=(0,2,1))
            u = E@E_T@(u.T.reshape((x.shape[1],dim,1)))
            u = u.reshape((x.shape[1],dim)).T
            u = u/np.linalg.norm(u, axis=0).reshape(1,-1)

            P_result['x'+str(P_index)][i] = temp_states
            # print(u.shape)
            temp_states = temp_states + alpha * u
            # if i> iter/2:
            Pi = Pi + alpha*np.linalg.norm(temp_states[:2,:]-self.target, axis=0)
        
        P_result['pi'+str(P_index)] = Pi
        

    def get_basis(self,n, e_num, orth = False ):
        #n ->Nxdim
        #e_directions -> e_numxNx3

        dim = n.shape[1]
        e_directions = np.zeros((e_num,n.shape[0],dim))
        temp = np.concatenate((n[:,1].reshape(n.shape[0],1), -n[:,0].reshape(n.shape[0],1)), axis=1)
        if dim == 2:
            e_directions[0,:,:] = temp
            if orth:
                return e_directions
            else:
                e_directions[1,:,:] = -temp
                return e_directions
        elif dim == 3:
            e_directions[0,:,:] = np.concatenate((temp,  np.zeros((n.shape[0],1))),axis =1)

        e0 = e_directions[0].reshape((n.shape[0],dim,1))
        if orth:
            r = R.from_rotvec(np.pi/2*n)
            Rot = r.as_matrix()
            e_directions[1] = (Rot@e0).reshape((n.shape[0],3))
            return e_directions[:,:dim]
        else:
            for i in range(e_num-1):
                r = R.from_rotvec(2*(i+1)*np.pi/e_num*n)
                Rot = r.as_matrix()
                e_directions[i+1] = (Rot@e0).reshape((n.shape[0],dim))
        return e_directions


    def geodesic_approx_phi(self,x, n, e_num, alpha_size, iter_num, checking_mode = False):
        # x->Nxdim
        # n ->Nxdim
        # return -> Nxdim
        dim = x.shape[1]
        shared_results = {}
        thread_list = []
        e_directions = self.get_basis(n/np.linalg.norm(n,axis=1).reshape(-1,1),e_num)
        # if dim ==3:
        #     e_directions = np.delete(e_directions, [0,int(e_num/2)], axis = 0)
        #     e_num = e_num-2

        # breakpoint()
        # shared_results['pi'+str(0)] = np.zeros(x.shape[0])
        # shared_results['x'+str(0)] = np.zeros((iter_num, dim,x.shape[0]))
        # self.receding_horizon(shared_results, 0, x.T, e_directions[0].T, alpha_size, iter_num)
        # exit()

        pi_list = np.zeros((e_num,x.shape[0]))
        for i in range(e_num):
            shared_results['pi'+str(i)] = np.zeros(x.shape[0])
            shared_results['x'+str(i)] = np.zeros((iter_num, dim,x.shape[0]))
            thread_list.append(threading.Thread(target=self.receding_horizon, args=(shared_results, i, x.T, e_directions[i].T, alpha_size, iter_num)))

        # Start threads
        for i in range(e_num):
            thread_list[i].start()

        # Wait for both threads to complete
        for i in range(e_num):
            thread_list[i].join()

        for i in range(e_num):
            pi_list[i] = shared_results['pi'+str(i)]

        # breakpoint()

        min_indices = np.argmin(pi_list, axis=0)

        if checking_mode:
            return shared_results, min_indices
        e_directions = np.moveaxis(e_directions, 1, 0)
        e_selected = e_directions[np.arange(x.shape[0])[:, None], min_indices.reshape(x.shape[0],1)]

        return e_selected.reshape((x.shape[0],dim))
    

    
    def h_grad_standard(self,x,normalize=True):
        #x -> Nxdim
        # breakpoint()
        dim = x.shape[1]
        if (not self.convex) and self.concave:
            dis, grad,_ = self.gpis.dis_normal_hes_func(x,normalize)
            return dis-1, grad, 0, True
        else:
            if self.concave:
                dis1, grad1 = self.h_gradc(x)
                dis2, grad2,_ = self.gpis.dis_normal_hes_func(x,normalize)
                dis2 = (0.5*(dis2-1)+1).reshape(x.shape[0],1)
                grad2 = grad2.reshape(x.shape[0],2)
                grad = np.concatenate((grad1, grad2),axis=0)
                dis = np.concatenate((dis1, dis2), axis=0)
            else:
                dis, grad = self.h_gradc(x)
            # breakpoint()
            grad_num = np.sum(np.exp(-self.rho*dis)*grad, axis=0).reshape(1,dim)
            
            grad_den = np.sum(np.exp(-self.rho*dis))
            dis_uni = -1/self.rho*np.log(grad_den)-1
            grad_uni = grad_num/grad_den

            gradt = self.h_grad_t(grad)
            # gradt_num = np.sum(np.exp(-self.rho*dis)*gradt)
            # gradt_uni = gradt_num/grad_den
            gradt_uni = self.compute_rel_vel(dis, gradt)
            return dis_uni, grad_uni, gradt_uni, (np.argmin(dis)>=self.xc.shape[0]).any()
    
    def h_grad_vector(self, x, normalize=True):
    #x -> nxd
    #return -> distance, gradient to x, gradient to t, is closest concave

        if (not self.convex) and self.concave:
            dis, grad = self.gpis.dis_normal_hes_func(x, normalize)
            # print(grad.shape)
            return dis-1, grad.T, 0, True
        else:
            if self.concave:
                dis1, grad1 = self.h_gradc(x)
                dis2, grad2= self.gpis.dis_normal_hes_func(x, normalize)
                dis2 = (0.5*(dis2-1)+1).reshape(1,x.shape[0])
                grad2 = grad2.T.reshape(1,grad2.shape[1], grad2.shape[0])
                dis = np.concatenate((dis1, dis2), axis=0)
                grad = np.concatenate((grad1, grad2),axis=0)
            else:
                dis, grad = self.h_gradc(x)
            
            grad_num = np.sum(np.exp(-self.rho*dis.reshape(dis.shape[0],dis.shape[1],1))*grad, axis=0)
            grad_den = np.sum(np.exp(-self.rho*dis),axis=0)
            dis_uni = -1/self.rho*np.log(grad_den)-1
            grad_uni = grad_num/grad_den.reshape(-1,1)

            # gradt = self.h_grad_t(grad)
            # gradt_num = -np.sum(np.exp(-self.rho*dis)*gradt, axis=0)
            # gradt_uni = gradt_num/grad_den.reshape(-1,1)
            # gradt_uni = self.compute_rel_vel(dis, gradt)
            return dis_uni, grad_uni, None, (np.argmin(dis)>=self.xc.shape[0]).any()
        
    def h_gradc(self,x):
        #x -> 1xdim
        #xc -> Nxdim
        #return -> Nxdim
        # breakpoint()
        if x.shape[0] !=1:
            diff = (-self.xc[:, np.newaxis] + x).reshape(-1, self.xc.shape[1])
            den = np.linalg.norm(diff, axis=1)
            dis = den.reshape(self.xc.shape[0],x.shape[0])-self.radius-self.margin+1
            grad = (diff/den.reshape(-1,1)).reshape(self.xc.shape[0],x.shape[0],2)
            return dis,grad
        den = np.linalg.norm(x-self.xc,axis=1).reshape(self.xc.shape[0],1)
        dis = den - self.radius -self.margin + 1
        grad = (x-self.xc)/den
        return dis, grad
    
    def h_grad_t(self, grad):
        #grad -> Nxdim or MxNxdim
        #xc -> Mxdim
        #return -> Nx1
        if not self.convex:
            return 0
        if self.concave:
            total_dxc = np.concatenate((self.dxc, np.zeros((1,self.dxc.shape[1]))), axis = 0)
        else:
            total_dxc = self.dxc
        if len(grad.shape)>2:
            return -np.sum(grad @ total_dxc.reshape(total_dxc.shape[0], total_dxc.shape[1],1), axis=2)
        else:
            return -np.sum(grad * total_dxc, axis=1)

    
    def update_obs(self):
        # breakpoint()
        # print("here")
        if self.t <=self.it_max: 
            self.t = self.t+1
            self.xc = self.xc+self.step
        return

    def update_xc(self, xc, dxc = None):
        #x -> Nxdim
        self.xc = xc
        # if (dxc!=None).all:
        #     self.dxc = dxc
        # else:
        self.dxc = np.zeros(self.xc.shape)
        return
    
    def compute_weights(self, dist,
        distMeas_lowerLimit=1,
        weightPow=2,
    ):
        """Compute weights based on a distance measure (with no upper limit)"""
        critical_points = dist <= distMeas_lowerLimit

        if np.sum(critical_points):  # at least one
            if np.sum(critical_points) == 1:
                w = critical_points * 1.0
                return w
            else:
                # TODO: continuous weighting function
                warnings.warn("Implement continuity of weighting function.")
                w = critical_points * 1.0 / np.sum(critical_points)
                return w


        dist = dist - distMeas_lowerLimit
        w = (1 / dist) ** weightPow
        if np.sum(w) == 0:
            return w
        w = w / np.sum(w)  # Normalization

        return w
    
    def compute_rel_vel(self, dis, gradt):
        # breakpoint()
        gradt = np.where(gradt<0, 0, gradt)
        weights = self.compute_weights(dis)

        return np.sum(weights*gradt)


    
    


    

 
