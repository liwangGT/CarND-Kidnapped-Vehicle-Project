/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double stdin[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
    num_particles = 10;
    default_random_engine generator;
    normal_distribution<double> x_gen(x, stdin[0]);
    normal_distribution<double> y_gen(y, stdin[1]);
    normal_distribution<double> theta_gen(theta, stdin[2]);
    for (int i =0; i<num_particles; i++){
    	weights.push_back(1.0);
    	Particle ptemp;
    	ptemp.x = x_gen(generator);
    	ptemp.y = y_gen(generator);
    	ptemp.theta = theta_gen(generator);
    	ptemp.weight = 1.0;
    	particles.push_back(ptemp);
    }

    // initialization finished
    is_initialized = true;
    cout<< "Initialization done!"<<endl;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
    for (int i =0; i<num_particles; i++){
    	default_random_engine generator;
	    normal_distribution<double> x_gen(0, std_pos[0]);
	    normal_distribution<double> y_gen(0, std_pos[1]);
	    normal_distribution<double> theta_gen(0, std_pos[2]);
        particles[i].x += velocity*cos(particles[i].theta)*delta_t + x_gen(generator);
		particles[i].y += velocity*sin(particles[i].theta)*delta_t + y_gen(generator);
		particles[i].theta += yaw_rate*delta_t + theta_gen(generator);
    }
    cout<<"Prediction step done!"<<endl;
}

void ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations,
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates
    
    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

double ParticleFilter::dataAssociation(const Map &map_landmarks, std::vector<LandmarkObs>& observations, double std_landmark[]) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
    double prob=1;
    double dist_min = 1e10;
    int j_min = -1;
    auto predicted = map_landmarks.landmark_list;
	for (int i =0; i<observations.size(); i++){
		dist_min = 1e10;
		j_min = -1;
		for (int j = 0; j<predicted.size(); j++){
			double dist = sqrt(pow(observations[i].x-predicted[j].x_f,2)
				             + pow(observations[i].y-predicted[j].y_f, 2));
			if (dist < dist_min){
				dist_min = dist;
				j_min = j;
			}
		}
        prob *= exp(-0.5*(std_landmark[0]*pow(observations[i].x-predicted[j_min].x_f,2) +
                          std_landmark[1]*pow(observations[i].y-predicted[j_min].y_f,2) ))/(2*M_PI*sqrt(std_landmark[0]*std_landmark[1]));
		observations[i].id = predicted[j_min].id_i;
		predicted.erase(predicted.begin()+j_min);
	}
    
    // change void to double, return probability
    return prob;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
    for (int i=0; i<num_particles; i++){
        // transform map_landmarks into local coordinates
        std::vector<LandmarkObs> obsG;
        cout<<"Convert to glabal"<<endl;
        Obs2Global(observations, particles[i], obsG);
        
        // match map_landmarks to observations, return probability as weights
        cout<<"calculate new weights"<<endl;
        particles[i].weight = dataAssociation(map_landmarks, obsG, std_landmark);
    }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
    random_device rd;
    mt19937 gen(rd());
    for (int i=0; i<num_particles; i++){
        weights[i] = particles[i].weight;
    }
    discrete_distribution<> d(weights.begin(), weights.end());
    vector<Particle> plist;
    for (int i=0; i<num_particles; i++){
        int ind = d(gen);
        plist.push_back(particles[ind]);
    }
    // overwrite list of particles with resampling
    particles = plist;

}

void ParticleFilter::Obs2Global(const std::vector<LandmarkObs> obsL, Particle &particle, std::vector<LandmarkObs>& obsG){
    std::vector<double> sx;
    std::vector<double> sy;
    std::vector<int> sa;
    for (int i =0; i<obsL.size(); i++){
        LandmarkObs temp;
        double ox = obsL[i].x;
        double oy = obsL[i].y;
        double ot = particle.theta;
        temp.x = particle.x + ox*cos(ot) - oy*sin(ot);
        temp.y = particle.y + ox*sin(ot) + oy*cos(ot);
        temp.id = obsL[i].id;
        obsG.push_back(temp);
        sx.push_back(temp.x);
        sy.push_back(temp.y);
        sa.push_back(temp.id);
    }
    SetAssociations(particle, sa, sx, sy);
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
