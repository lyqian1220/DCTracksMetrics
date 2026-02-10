import pybes3 as p3

from src.match.matcher import Match

def total_track_metrics(evt, mc_dict, rec_dict):
    """
    Calculate total track metrics including efficiency and purity for each track index 
    """
    
    num_TotalRecHits = Match.get_nTotRecHits(evt)

    all_list = []
    for i in range(len(mc_dict)):
        event_dict = {}

        if not rec_dict[i] or not mc_dict[i]:
            all_list.append(event_dict)
            continue

        # Iterate over all reconstructed tracks in current event
        for trkid, track_data in rec_dict[i].items():

            if trkid not in num_TotalRecHits[i]:
                continue

            nTotalRecHits = num_TotalRecHits[i][trkid]
            event_dict[trkid] = {"Rec": {}}

            rec_hits = track_data.get("Rec", {})
            # Calculate metrics for each track index
            for trackIndex, hit_info in rec_hits.items():
                # Number of shared hits between reconstructed and MC truth
                nSharedHits = hit_info.get("nHits", 0)
                purity = nSharedHits / nTotalRecHits

                # Calculate efficiency if corresponding MC track exists
                if trackIndex in mc_dict[i]:
                    nTruthHits = mc_dict[i][trackIndex].get("nHits", 0)
                    eff = nSharedHits / nTruthHits
                else:
                    # Mark efficiency as -1 for non-matching MC track
                    eff = -1

                event_dict[trkid]["Rec"][trackIndex] = {
                    "nHits": nSharedHits,
                    "eff": eff,
                    "purity": purity
                }

        all_list.append(event_dict)
    
    return all_list

def matched_and_fake_track_metrics(evt, mc_dict, rec_dict):
    """
    Classify tracks into matched and fake categories
    Args:
        evt: Event data container
        mc_dict: Dictionary of MC truth
        rec_dict: Dictionary of reconstructed track data
    """
    # Get base track metrics from total_track_metrics
    metrics = total_track_metrics(evt, mc_dict, rec_dict)
    matched_dict = []
    fake_dict = []
    fake_count = 0
    
    for i, evt_dict in enumerate(metrics):
        evt_matched ={}
        evt_fake = {}
        
        if not evt_dict or not mc_dict[i]:
            matched_dict.append(evt_matched)
            fake_dict.append(evt_fake)
            continue
        
        # Get valid MC track indices for current event
        valid_mc_indices = set(mc_dict[i].keys())

        for trkid, trk_data in evt_dict.items():
            rec_dict = trk_data.get('Rec', {})
            track_indices = list(rec_dict.keys())

            # Classify as fake if all track indices are -1
            all_negative_one = all(idx == -1 for idx in track_indices)
            if all_negative_one:
                fake_count += 1
                evt_fake[trkid] = trk_data
                continue
            
            # Process tracks with single index
            if len(track_indices) == 1:
                selected_idx = track_indices[0]
                # Classify as matched if index exists in MC data
                if selected_idx in valid_mc_indices:
                    evt_matched[trkid] = {
                        'Rec': {selected_idx: rec_dict[selected_idx]}
                    }
                
            else:
                print(f"Multiple indices!!!.")

                # Filter indices by purity (>0.5) and minimum hit count (>=6)
                qualified_indices = [
                    idx for idx in track_indices 
                    if rec_dict[idx].get('purity', 0) > 0.5 and rec_dict[idx].get('nHits', 0) >= 6
                ]

                if not qualified_indices:
                    continue
                
                if qualified_indices[0] in valid_mc_indices:
                    evt_matched[trkid] = {
                        'Rec': {qualified_indices[0]: rec_dict[qualified_indices[0]]}
                    }

        matched_dict.append(evt_matched)
        fake_dict.append(evt_fake)
    return matched_dict, fake_dict, fake_count

    
def filter_mc_by_matched_tracks(evt, mc_dict, matched_dict):
    """
    Filter MC track data to keep only tracks that have matched reconstructed tracks
    Args:
        evt: Event data container
        mc_dict: Original MC track dictionary
        matched_dict: Matched reconstructed tracks dictionary
    """
    filtered_mc_dict = []
    
    for evt_idx, (evt_matched, evt_mc) in enumerate(zip(matched_dict, mc_dict)):
        if not evt_matched or not evt_mc:
            print("filter_mc_by_matched_tracks->no matched tracks or no mc tracks!!!")
            filtered_mc_dict.append({})
            continue

        current_filtered_mc = {}
        matched_track_indices = set()
        # Collect all MC track indices from matched reconstructed tracks
        for trkid, trk_data in evt_matched.items():
            rec_info = trk_data.get('Rec', {})
            track_indices = list(rec_info.keys())
            matched_track_indices.update(track_indices)
        
        # Check for invalid indices (not present in MC data)
        has_invalid_index = False
        for track_idx in matched_track_indices:
            if track_idx not in evt_mc:
                has_invalid_index = True
                break
        
        # Append empty dict if invalid indices found, else append filtered MC data
        if has_invalid_index:
            filtered_mc_dict.append({})
        else:
            filtered_mc_dict.append(evt_mc)

    return filtered_mc_dict

def matched_and_clone_track_metrics(first_matched_dict):
    """
    Identify clone tracks (multiple reconstructions for same MC track) and keep best candidate
    """
    updated_matched_dict = []
    clone_track_dict = []
    clone_count = 0 
    
    for i, evt_dict in enumerate(first_matched_dict):
        evt_updated_matched_dict = {}
        evt_clone_track_dict = {}

        if not evt_dict:
            updated_matched_dict.append(evt_updated_matched_dict)
            clone_track_dict.append(evt_clone_track_dict)
            continue
        
        # Group tracks by their MC track index
        tracks_by_index = {}
        for trkid, trk_data in evt_dict.items():
            rec_dict = trk_data.get('Rec', {})
            trackIndex = next(iter(rec_dict.keys())) 
            
            if trackIndex not in tracks_by_index:
                tracks_by_index[trackIndex] = []
            tracks_by_index[trackIndex].append({
                "trackId": trkid,
                "track": trk_data  
            })

        # Process each group of tracks sharing the same MC index
        for trackIndex, track_candidates in tracks_by_index.items():
            # Single candidate - keep as matched track
            if len(track_candidates) == 1:
                trkid = track_candidates[0]["trackId"]
                evt_updated_matched_dict[trkid] = track_candidates[0]["track"]
            # Multiple candidates - identify clones and select best candidate
            else:
                # Select best candidate by efficiency (primary) and purity (secondary)
                best_candidate = max(
                    track_candidates,
                    key=lambda x: (
                        x["track"]["Rec"][trackIndex]["eff"],
                        x["track"]["Rec"][trackIndex]["purity"]
                    )
                )
                best_trkid = best_candidate["trackId"]
                evt_updated_matched_dict[best_trkid] = best_candidate["track"]
                
                # Classify remaining candidates as clones
                for candidate in track_candidates:
                    if candidate["trackId"] != best_trkid:
                        trkid = candidate["trackId"]
                        clone_count += 1
                        evt_clone_track_dict[trkid] = candidate["track"]
        
        updated_matched_dict.append(evt_updated_matched_dict)
        clone_track_dict.append(evt_clone_track_dict)
    
    return updated_matched_dict, clone_track_dict, clone_count

def good_and_fake_track_metrics(evt, updated_matched_dict, fake_track_dict, fake_count, filtered_mc_dict_with_info, store_status: bool = False):
    """
    Classify matched tracks as good (efficiency > 0.2) or fake, and include track status if requested
    """
    # Get reconstruction status data if storage is enabled
    if store_status:
        recMdcKalTrack = evt["TRecEvent/m_recMdcKalTrackCol"].array()

    good_count = 0
    good_track_dict = []
    all_matched_dict = []
    
    for i, evt_dict in enumerate(updated_matched_dict):
        evt_good_track_dict = {}

        if not evt_dict:
            good_track_dict.append(evt_good_track_dict)
            continue
        
        if store_status:
            stat = recMdcKalTrack[i].m_stat

        for trkid, trk_data in evt_dict.items():
            rec_dict = trk_data.get('Rec', {})
            trackIndex = next(iter(rec_dict.keys()))
            metrics_data = rec_dict[trackIndex]

            nhits = metrics_data["nHits"]
            eff = metrics_data["eff"]
            purity = metrics_data["purity"]

            pid = filtered_mc_dict_with_info[i][trackIndex]["pid"]
            pid_mapping_loc = Match.PID_TO_INDEX_MAPPING.get(pid, 2)

            track_rec_info = {
                'Rec': {
                    trackIndex: {"nHits": nhits, "eff": eff, "purity": purity}
                }
            }

            # Add status information if requested
            if store_status:
                status = stat[trkid][0][pid_mapping_loc]
                track_rec_info["status"] = int(status)
                
            # Classify as good track if efficiency > 0.2
            if eff > 0.2 :
                good_count += 1
                evt_good_track_dict[trkid] = track_rec_info
            # Reclassify as fake track if efficiency <= 0.2
            else:
                fake_count += 1
                fake_track_dict[i][trkid] = track_rec_info

        good_track_dict.append(evt_good_track_dict)
        all_matched_dict.append(track_rec_info)

    return good_track_dict, fake_track_dict, good_count, fake_count, all_matched_dict