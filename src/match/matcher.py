import pybes3 as p3
from collections import defaultdict
import numpy as np
from particle import PDGID


class Match:
    # Valid particle IDs for 5 kinds of particles (e, μ, π, K, p) and their antiparticles
    VALID_PIDS = {11, -11, 13, -13, 211, -211, 321, -321, 2212, -2212}

    # Map PDG IDs to simplified index (for fitting purposes)
    PID_TO_INDEX_MAPPING = {
        11: 0, -11: 0,  # Electron (e+) / Positron (e-) -> index 0
        13: 1, -13: 1,  # Muon (μ+) / Anti-muon (μ-) -> index 1
        211: 2, -211: 2, # Pion (π+) / Anti-pion (π-) -> index 2
        321: 3, -321: 3, # Kaon (K+) / Anti-kaon (K-) -> index 3
        2212: 4, -2212: 4 # Proton (p) / Anti-proton (p-bar) -> index 4
    }

    @staticmethod
    def compute_kinematics(momX, momY, momZ):
        """
        Compute key kinematic variables from 3-momentum components
        Args:
            momX: X-component of momentum (float/array)
            momY: Y-component of momentum (float/array)
            momZ: Z-component of momentum (float/array)
        Returns:
            tuple: total momentum (p), transverse momentum (pT), cos(theta), azimuthal angle (phi)
        """
        pT = np.sqrt(momX**2 + momY**2)  # Transverse momentum (perpendicular to beam axis)
        p = np.sqrt(momX**2 + momY**2 + momZ**2)  # Total momentum
        cos = np.where(p > 0, momZ / p, 0)  # Cosine of polar angle (theta)
        phi = np.arctan2(momY, momX)  # Azimuthal angle (phi)
        return p, pT, cos, phi

    @staticmethod
    def create_helix(posX, posY, posZ, momX, momY, momZ, charge):
        """
        Create helix object for particle track (using pybes3)
        Args:
            posX/Y/Z: Initial position coordinates (float)
            momX/Y/Z: Initial momentum components (float)
            charge: Particle charge (int, from PDG ID)
        Returns:
            pybes3 helix_obj: Helix track model with pivot at (0,0,0)
        """
        return p3.helix_obj(
            position=(float(posX), float(posY), float(posZ)),
            momentum=(float(momX), float(momY), float(momZ)),
            pivot=(float(posX), float(posY), float(posZ)),
            charge=int(charge),
        ).change_pivot(0,0,0)  # Reset pivot to origin


    def get_mc_dict(evt: any, get_info: bool = True, if_phy: bool = False):
        """
        Extract Monte Carlo (MC) truth information and map to reconstructed track IDs
        Args:
            evt: Event data container
            get_info: Whether to include detailed kinematic info (bool)
            if_phy: Whether the events are physics process, if it is, skip physics selection cuts
        """

        mcParticle = evt["TMcEvent/m_mcParticleCol"].array() 
        mdcDigiCol = evt["TDigiEvent/m_mdcDigiCol"].array() 
        trackIndex = mdcDigiCol.m_trackIndex  
        event = evt["TEvtHeader/m_eventId"].array()  
        run = evt["TEvtHeader/m_runId"].array()   

        mc_fields = ["m_xInitialMomentum", "m_yInitialMomentum", "m_zInitialMomentum",
                    "m_xInitialPosition", "m_yInitialPosition", "m_zInitialPosition",
                    "m_particleID"]

        result_list = []
        for i, (evt_mc, evt_trackIndex) in enumerate(zip(mcParticle, trackIndex)):
            if len(evt_mc) == 0 or len(evt_trackIndex) == 0:
                result_list.append({})
                continue
            
            # Parse track index: split into truth ID (mod 10000) and track ID (div 10000)
            idx_array = np.array(evt_trackIndex)
            idx_array = idx_array[idx_array > 0]  
            truthID_idx = idx_array % 10000      
            trackID_idx = idx_array // 10000     

            # Count unique (truth ID, track ID) pairs (hit counts)
            combined = np.column_stack((truthID_idx, trackID_idx))
            unique_combined, counts = np.unique(combined, axis=0, return_counts=True)
            result = np.column_stack((unique_combined, counts))

            pid = evt_mc.m_particleID.to_numpy()
            base_result = {}
            for item in result:
                truth_id = int(item[0])
                track_id = int(item[1])
                count = int(item[2])
                
                pid_val = int(pid[truth_id])
                base_result[track_id] = {"nHits": count, "pid": pid_val, "truth_id": truth_id}
            
            if not get_info:
                result_list.append(base_result)
                continue
            
            try:
                mc_indices = result[:, 0].astype(int)
                mc_data = {
                    field: getattr(evt_mc, field).to_numpy()[mc_indices] 
                    for field in mc_fields
                }
            except IndexError: 
                result_list.append(base_result)
                continue
            
            # Compute kinematic variables for MC particles
            m_p, m_pT, m_cos, m_phi = Match.compute_kinematics(
                mc_data["m_xInitialMomentum"],
                mc_data["m_yInitialMomentum"],
                mc_data["m_zInitialMomentum"]
            )
            
            # Get charge from PDG ID (particle package)
            m_charge = [int(PDGID(pid_val).charge) for pid_val in mc_data["m_particleID"]]
            # Calculate transverse radius (initial position)
            radius = np.sqrt(mc_data["m_xInitialPosition"]**2 + mc_data["m_yInitialPosition"]**2) 
            posX, posY, posZ = mc_data["m_xInitialPosition"], mc_data["m_yInitialPosition"], mc_data["m_zInitialPosition"]
            momX, momY, momZ = mc_data["m_xInitialMomentum"], mc_data["m_yInitialMomentum"], mc_data["m_zInitialMomentum"]
            
            # Add helix and kinematic info to each track
            for i, item in enumerate(result):
                track_id = int(item[1])
                helix = Match.create_helix(
                    posX[i], posY[i], posZ[i],
                    momX[i], momY[i], momZ[i],
                    m_charge[i]
                )

                # Store detailed MC info (kinematics + helix parameters)
                base_result[track_id]["Mc"] = {
                    "px": float(momX[i]),          # X momentum
                    "py": float(momY[i]),          # Y momentum
                    "pz": float(momZ[i]),          # Z momentum
                    "p": float(m_p[i]),            # Total momentum
                    "pT": float(m_pT[i]),          # Transverse momentum
                    "cos": float(m_cos[i]),        # cos(theta)
                    "phi": float(m_phi[i]),        # Azimuthal angle
                    "radius": float(radius[i]),    # Transverse radius (initial position)
                    "charge": int(m_charge[i]),    # Particle charge
                    "dr": helix.dr,                # Helix parameter: dR
                    "phi0": helix.phi0,            # Helix parameter: phi0
                    "kappa": helix.kappa,          # Helix parameter: kappa 
                    "dz": helix.dz,                # Helix parameter: dZ
                    "tanl": helix.tanl             # Helix parameter: tan(lambda)
                }

            # Apply physics selection cuts (filter tracks by kinematics)
            if not if_phy:
                tracks_to_remove = []
                for track_id, info in base_result.items():
                    mc_info = info.get("Mc")
                    if mc_info is None:
                        continue
                    # Selection cuts: cos(theta) in [-0.93, 0.93], pT in [0.15, 1.5] GeV/c
                    cos_val = mc_info["cos"]
                    pt_val = mc_info["pT"]
                    if not (-0.93 <= cos_val <= 0.93 and 0.15 <= pt_val <= 1.5):
                        print(f"Removing TrackID:{track_id} with cos:{cos_val}, pT:{pt_val}")
                        tracks_to_remove.append(track_id)
                
                for track_id in tracks_to_remove:
                    del base_result[track_id]

            if not base_result:
                result_list.append({})
            else:
                result_list.append(base_result)

        return result_list

    def get_rec_dict(evt):
        """
        Extract reconstructed track information and map to MDC digi hits
        """

        mdcDigiCol = evt["TDigiEvent/m_mdcDigiCol"].array()
        mcParticle = evt["TMcEvent/m_mcParticleCol"].array()

        event = evt["TEvtHeader/m_eventId"].array()
        run = evt["TEvtHeader/m_runId"].array()
 
        trackIndex = mdcDigiCol.m_trackIndex 
        mdc_digi = p3.parse_mdc_digi(mdcDigiCol) 
        
        trackIndexOfDigiDict = [] 
        for i, (evt_gids, evt_idx, mc) in enumerate(zip(mdc_digi['gid'], trackIndex, mcParticle)):
            idx_array = np.array(evt_idx)
            idx_array = np.where(idx_array < 0, -1, idx_array)  # Mark noise indices as -1
            idx_array = np.where(idx_array > 0, (idx_array // 10000).round().astype(int), idx_array)  # Extract track ID
            
            event_dict = dict(zip(evt_gids, idx_array))
            trackIndexOfDigiDict.append(event_dict)
        
        # Extract reconstructed MDC hits and map to track IDs
        recMdcHitCol = evt["TRecEvent/m_recMdcHitCol"].array()
        mdcId = recMdcHitCol.m_mdcid
        mdc_digi = p3.parse_mdc_digi_id(mdcId) 
        hit_trkId = recMdcHitCol.m_trkid 

        # Map track IDs to list of digi GIDs (per event)
        digiXtrk = [] 
        for evt_gids, evt_hit_trkId in zip(mdc_digi['gid'], hit_trkId):
            event_dict = defaultdict(list)
            for gid, trkid in zip(evt_gids, evt_hit_trkId):
                event_dict[int(trkid)].append(int(gid))
            digiXtrk.append(event_dict)
        
        recMdcTrackCol = evt["TRecEvent/m_recMdcTrackCol"].array()
        trackId = recMdcTrackCol.m_trackId

        rec_dict = []
        for i, (evt_gids, evt_trackId, current_gid_map) in enumerate(zip(digiXtrk, trackId, trackIndexOfDigiDict)):
            event_dict = {}
            if len(evt_trackId)==0:
                rec_dict.append(event_dict)
                continue
            
            track_ids = [int(trkid) for trkid in evt_trackId]
            for trkid_int in track_ids:
                gids = evt_gids.get(trkid_int, []) 

                rec_counts = defaultdict(int)
                for gid_int in gids:
                    if gid_int in current_gid_map:
                        track_index = current_gid_map[gid_int]
                        rec_counts[track_index] += 1
                
                # Store hit counts for valid tracks
                if rec_counts:
                    event_dict[trkid_int] = {
                        "Rec": {int(idx): {"nHits": cnt} for idx, cnt in rec_counts.items()}
                    }

            rec_dict.append(event_dict)
        
        return rec_dict
    
    def get_recTrkMatchedWithMdcInfo(evt, rec_dict):
        """
        Results of finding, add kinematic info to reconstructed tracks
        Args:
            evt: Event data container
            rec_dict: Reconstructed track dict (from get_rec_dict)
        """
        recMdcTrackCol = evt["TRecEvent/m_recMdcTrackCol"].array()
        event = evt["TEvtHeader/m_eventId"].array()
        run = evt["TEvtHeader/m_runId"].array()
        
        for i, (evt_recTrk_dict, evt_mdc_dict) in enumerate(zip(rec_dict, recMdcTrackCol)):

            if not evt_recTrk_dict:
                continue
            
            # Extract helix and kinematic data from MDC tracks
            evt_helix = evt_mdc_dict.m_helix  # Helix parameters (dr, phi0, kappa, dz, tanl)
            kinematics = p3.parse_helix(evt_helix)  # Parse helix to get momentum/position
            
            px = kinematics.px 
            py = kinematics.py
            pz = kinematics.pz
            x = kinematics.x 
            y = kinematics.y
            radius = np.sqrt(x ** 2 + y ** 2) 
            charge = kinematics.charge
            helix_dr = evt_helix[:, 0] 
            helix_phi0 = evt_helix[:, 1]
            helix_kappa = evt_helix[:, 2]
            helix_dz = evt_helix[:, 3]
            helix_tanl = evt_helix[:, 4]
            
            p, pT, cos, phi = Match.compute_kinematics(px, py, pz)
            
            track_items = evt_recTrk_dict.items()
            for trkid, track_info in track_items:
                trkid_int = int(trkid) 
                
                # Store kinematic parameters
                track_info["Mdc"] = {
                    "px": float(px[trkid_int]),
                    "py": float(py[trkid_int]),
                    "pz": float(pz[trkid_int]),
                    "pT": float(pT[trkid_int]),
                    "p": float(p[trkid_int]),
                    "cos": float(cos[trkid_int]),
                    "phi": float(phi[trkid_int]),
                    "radius": float(radius[trkid_int]),
                    "charge": int(charge[trkid_int]),
                    "dr": float(helix_dr[trkid_int]),
                    "phi0": float(helix_phi0[trkid_int]),
                    "kappa": float(helix_kappa[trkid_int]),
                    "dz": float(helix_dz[trkid_int]),
                    "tanl": float(helix_tanl[trkid_int]),
                }
        return rec_dict

    def get_recTrkMatchedWithKalInfo(evt, mc_dict, rec_dict):
        """
        Results of fitting, add Kalman track info to reconstructed tracks 
        """
        recMdcKalTrackCol = evt["TRecEvent/m_recMdcKalTrackCol"].array() 
        event = evt["TEvtHeader/m_eventId"].array()
        run = evt["TEvtHeader/m_runId"].array()

        for i, (evt_recTrk_dict, evt_kal_dict, evt_mc_dict) in enumerate(zip(
            rec_dict, recMdcKalTrackCol, mc_dict
        )):
            if not evt_recTrk_dict:
                continue

            evt_stat = evt_kal_dict.m_stat  # whether pass the fitting, "0" - fit successfully
            evt_helix = evt_kal_dict.m_zhelix_e  # Kalman helix parameters
            evt_trkid = evt_kal_dict.m_trackId  # Track IDs for Kalman tracks

            # Parse helix to get kinematics
            kinematics = p3.parse_helix(evt_helix)
            charge = kinematics.charge
            px = kinematics.px
            py = kinematics.py
            pz = kinematics.pz
            
            p, pT, _, _ = Match.compute_kinematics(px, py, pz)
            
            track_items = evt_recTrk_dict.items()
            trkid_to_index = {trk: idx for idx, trk in enumerate(evt_trkid)}
            
            # Add Kalman info to each track
            for trkid, track_info in track_items:
                index = trkid_to_index[trkid]
                status = 1  # Default status (not pass the fitting)
                
                if evt_mc_dict:
                    first_mc = next(iter(evt_mc_dict.values()))
                    pid = first_mc["pid"]
                    pid_mapping_loc = Match.PID_TO_INDEX_MAPPING.get(pid, 2)  # Default to pion if unknown
                    status = int(evt_stat[index][0][pid_mapping_loc])
                else:
                    if 0 in evt_stat[index][0]:
                        status = 0  # Mark invalid if status 0 present

                # Store fitting info
                track_info["Kal"] = {
                    "status": status,          # Fitting status
                    "px": float(px[index]),   
                    "py": float(py[index]),    
                    "pz": float(pz[index]),    
                    "pT": float(pT[index]),    
                    "p": float(p[index]),      
                    "charge": int(charge[index]),  
                    "dr": float(evt_helix[index][0]),    
                    "phi0": float(evt_helix[index][1]),  
                    "kappa": float(evt_helix[index][2]), 
                    "dz": float(evt_helix[index][3]),    
                    "tanl": float(evt_helix[index][4]),  
                }

        return rec_dict
    
    def get_nTotRecHits(evt):
        """
        Get total number of reconstructed hits per track
        """
        recMdcTrackCol = evt["TRecEvent/m_recMdcTrackCol"].array()
        trackId = recMdcTrackCol.m_trackId       
        nRecHits = recMdcTrackCol.m_nhits       
        
        nTotRecHits = []
        for i, (evt_trackId, evt_nRecHits) in enumerate(zip(trackId, nRecHits)):
            event_dict = {}

            if len(evt_trackId) == 0 or len(evt_nRecHits) == 0:
                nTotRecHits.append(event_dict)
                continue

            trk_ids = np.array(evt_trackId)
            hits = np.array(evt_nRecHits)

            for trk_id, hit in zip(trk_ids, hits):
                event_dict[int(trk_id)] = int(hit)
            
            nTotRecHits.append(event_dict)
        
        return nTotRecHits