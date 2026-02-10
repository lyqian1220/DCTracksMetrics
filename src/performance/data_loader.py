from src.match import metrics
from src.match.matcher import Match
from src.performance import hit
from src.performance import track
from src.performance.initializer import cache_results, load_evt
import uproot
uproot.default_library = "np"

def load_evt_dict(evt_file, evtlabel, cache_path):
    evt_cache_key = f"evt_dict_{evtlabel}"
    evt = cache_results(
        load_evt, 
        evt_file,
        cache_key=evt_cache_key,
        cache_path=cache_path,
    )
    return evt

def load_mc_dict(evt, get_info, if_phy, evtlabel, cache_path):
    mc_cache_key = f"mc_dict_{evtlabel}"
    mc_dict = cache_results(
        Match.get_mc_dict,
        evt,
        get_info=get_info,
        if_phy=if_phy,
        cache_path=cache_path,
        cache_key=mc_cache_key
    )
    return mc_dict

def load_rec_dict(evt, evtlabel, cache_path):
    rec_cache_key = f"rec_dict_{evtlabel}"
    rec_dict = cache_results(
        Match.get_rec_dict,
        evt,
        cache_path=cache_path,
        cache_key=rec_cache_key
    )
    return rec_dict

def load_mdc_tracks_info(evt, rec_dict, evtlabel, cache_path):
    cache_key = f"Mdc_dict_{evtlabel}"  
    track_mdc_dict = cache_results(
        Match.get_recTrkMatchedWithMdcInfo,
        evt,
        rec_dict,
        cache_path=cache_path,
        cache_key=cache_key
    )
    return track_mdc_dict

def load_kal_tracks_info(evt, mc_dict, rec_dict,  evtlabel, cache_path):
    cache_key = f"Kal_dict_{evtlabel}" 
    track_kal_dict = cache_results(
        Match.get_recTrkMatchedWithKalInfo,
        evt,
        mc_dict,
        rec_dict,
        cache_path=cache_path,
        cache_key=cache_key
    )
    return track_kal_dict


def load_matched_and_fake_track_metrics(evt, mc_dict, rec_dict, evtlabel, cache_path=None):
    cache_key = f"matched_and_fake_track_metrics_{evtlabel}"
    matched_dict, fake_dict, fake_count = cache_results(
        metrics.matched_and_fake_track_metrics,
        evt, mc_dict, rec_dict,
        cache_path=cache_path,
        cache_key=cache_key
    )
    return matched_dict, fake_dict, fake_count

def load_filter_mc_by_matched_tracks(evt, mc_dict, matched_dict, evtlabel, cache_path=None):
    cache_key = f"filter_mc_by_matched_tracks_{evtlabel}"
    filtered_mc_dict = cache_results(
        metrics.filter_mc_by_matched_tracks,
        evt, mc_dict, matched_dict,
        cache_path=cache_path,
        cache_key=cache_key
    )
    return filtered_mc_dict

def load_matched_and_clone_track_metrics(matched_dict, evtlabel, cache_path=None):

    cache_key = f"matched_and_clone_track_metrics_{evtlabel}"
    updated_matched_dict, clone_dict, clone_count = cache_results(
        metrics.matched_and_clone_track_metrics,
        matched_dict,
        cache_path=cache_path,
        cache_key=cache_key
    )
    return updated_matched_dict, clone_dict, clone_count

def load_good_and_fake_tracks(evt, updated_matched_dict, fake_track_dict, fake_count, filtered_mc_dict_with_info, store_status=False, evtlabel="trad", cache_path=None):
    cache_key = f"good_and_fake_track_metrics_{evtlabel}_status_{store_status}"
    good_track_dict, fake_track_dict, good_count, fake_count, all_matched_dict = cache_results(
        metrics.good_and_fake_track_metrics,
        evt, 
        updated_matched_dict, 
        fake_track_dict, 
        fake_count,
        filtered_mc_dict_with_info, 
        store_status=store_status,
        cache_path=cache_path,
        cache_key=cache_key
    )
    
    return good_track_dict, fake_track_dict, good_count, fake_count, all_matched_dict


def load_common_mc_dict(mc_dict_trad, mc_dict_ml, cache_path):
    mc_cache_key = f"common_mc_dict"
    common_mc_dict = cache_results(
        track.extract_common_mc_info,
        mc_dict_trad, 
        mc_dict_ml,
        cache_path=cache_path,
        cache_key=mc_cache_key
    )
    return common_mc_dict

def load_hit_violin_data(common_mc_dict, good_track_dict_trad, good_track_dict_ml, mdc_track_dict_trad, mdc_track_dict_ml, cache_path=None):
    
    cache_key = "hit_violin_data"
    hit_violin_data = cache_results(
        hit.get_hit_violin_data,
        common_mc_dict=common_mc_dict,
        good_track_dict_trad=good_track_dict_trad,
        good_track_dict_ml=good_track_dict_ml,
        mdc_track_dict_trad=mdc_track_dict_trad,
        mdc_track_dict_ml=mdc_track_dict_ml,
        cache_path=cache_path,
        cache_key=cache_key
    )
    return hit_violin_data

def load_all_data(file_trad, file_ml, isMDC=False, isPhy=False, cache_path=None):
    if not isMDC:
        store_status=True
    else:
        store_status=False

    base_evtlabel_trad = "trad"
    base_evtlabel_ml = "ml"
    if isMDC:
        evtlabel_trad = f"{base_evtlabel_trad}_MDC" 
        evtlabel_ml = f"{base_evtlabel_ml}_MDC"   
    else:
        evtlabel_trad = base_evtlabel_trad
        evtlabel_ml = base_evtlabel_ml


    evtTrad = load_evt(file_trad)
    evtML = load_evt(file_ml)


    mc_dict_trad = load_mc_dict(evtTrad, get_info=True, if_phy=isPhy, evtlabel=evtlabel_trad, cache_path=cache_path)
    mc_dict_ml = load_mc_dict(evtML, get_info=True, if_phy=isPhy, evtlabel=evtlabel_ml, cache_path=cache_path)

    rec_dict_trad = load_rec_dict(evtTrad, evtlabel_trad, cache_path)
    rec_dict_ml = load_rec_dict(evtML, evtlabel_ml, cache_path)
    
    track_mdc_dict_trad = load_mdc_tracks_info(evtTrad, rec_dict_trad, evtlabel=evtlabel_trad, cache_path=cache_path)
    track_mdc_dict_ml = load_mdc_tracks_info(evtML, rec_dict_ml, evtlabel=evtlabel_ml, cache_path=cache_path)

    matched_dict_trad, fake_dict_trad, fake_count_trad = load_matched_and_fake_track_metrics(
        evtTrad, 
        mc_dict_trad,
        rec_dict_trad, 
        evtlabel=evtlabel_trad, 
        cache_path=cache_path
    )
    matched_dict_ml, fake_dict_ml, fake_count_ml = load_matched_and_fake_track_metrics(
        evtML, 
        mc_dict_ml, 
        rec_dict_ml, 
        evtlabel=evtlabel_ml, 
        cache_path=cache_path
    )


    filtered_mc_dict_trad = load_filter_mc_by_matched_tracks(
        evtTrad, 
        mc_dict_trad,
        matched_dict_trad,  
        evtlabel=evtlabel_trad, 
        cache_path=cache_path
    )
    filtered_mc_dict_ml = load_filter_mc_by_matched_tracks(
        evtML, 
        mc_dict_ml,
        matched_dict_ml, 
        evtlabel=evtlabel_ml, 
        cache_path=cache_path
    )
    
    

    updated_matched_dict_trad, clone_dict_trad, clone_count_trad = load_matched_and_clone_track_metrics(matched_dict_trad, evtlabel=evtlabel_trad, cache_path=cache_path)
    updated_matched_dict_ml, clone_dict_ml, clone_count_ml = load_matched_and_clone_track_metrics(matched_dict_ml, evtlabel=evtlabel_ml, cache_path=cache_path)

    good_track_dict_trad, fake_track_dict_all_trad, good_count_trad, fake_count_all_trad, all_matched_dict_trad = load_good_and_fake_tracks(
        evtTrad, 
        updated_matched_dict_trad, 
        fake_dict_trad,
        fake_count_trad, 
        filtered_mc_dict_trad,
        store_status=store_status, 
        evtlabel=evtlabel_trad, 
        cache_path=cache_path
    )

    good_track_dict_ml, fake_track_dict_all_ml, good_count_ml, fake_count_all_ml, all_matched_dict_ml = load_good_and_fake_tracks(
        evtML, 
        updated_matched_dict_ml, 
        fake_dict_ml,
        fake_count_ml, 
        filtered_mc_dict_ml,
        store_status=store_status, 
        evtlabel=evtlabel_ml, 
        cache_path=cache_path
    )

    common_mc_dict = load_common_mc_dict(
        mc_dict_trad=filtered_mc_dict_trad,
        mc_dict_ml=filtered_mc_dict_ml,
        cache_path=cache_path
    )

    base_data = {
        "mc_dict": {
            "trad": mc_dict_trad,
            "ml": mc_dict_ml,
        },
        "mc_dict_filtered": {
            "trad": filtered_mc_dict_trad, 
            "ml": filtered_mc_dict_ml,
            "common": common_mc_dict
        },
        "track_info_dict": {
            "Mdc": {"trad": track_mdc_dict_trad, "ml": track_mdc_dict_ml},
        },
        "good_track_dict": {
            "trad": good_track_dict_trad,
            "ml": good_track_dict_ml,
        },
        
    }
    
    if not isMDC:
        track_kal_dict_trad = load_kal_tracks_info(evtTrad, mc_dict_trad, rec_dict_trad, evtlabel=evtlabel_trad, cache_path=cache_path)
        track_kal_dict_ml = load_kal_tracks_info(evtML, mc_dict_ml, rec_dict_ml, evtlabel=evtlabel_ml, cache_path=cache_path)

        hit_violin_data_all = load_hit_violin_data(
            common_mc_dict=common_mc_dict,
            good_track_dict_trad=good_track_dict_trad,
            good_track_dict_ml=good_track_dict_ml,
            mdc_track_dict_trad=track_mdc_dict_trad,
            mdc_track_dict_ml=track_mdc_dict_ml,
            cache_path=cache_path
        )
        
        return {
            **base_data,
            "track_info_dict": {
                "Mdc": base_data["track_info_dict"]["Mdc"],
                "Kal": {"trad": track_kal_dict_trad, "ml": track_kal_dict_ml}
            },
            "fake_track_dict": {
                "trad": fake_track_dict_all_trad, 
                "ml": fake_track_dict_all_ml,
            },
            "clone_track_dict": {
                "trad": clone_dict_trad, 
                "ml": clone_dict_ml
            },
            "hit_violin_data": hit_violin_data_all,
            "all_matched_dict": {
                "trad": all_matched_dict_trad,
                "ml": all_matched_dict_ml,
            },
        }
    else:
        return base_data