/* BridgeAI: Reinforcement Learning Bridge for the Spring/Recoil engine.
 *
 * Runs as a Skirmish AI on team 0 (the agent side). Every decision_interval frames,
 * it collects observations from all units (own + visible enemies), sends them to Python
 * over a TCP socket (blocking), receives actions, and gives orders directly via the
 * engine callback — no server-lead delay, exact gymnasium loop.
 *
 * For episode resets, it calls the synced gadget (reinforcement_learning_scenario.lua)
 * via COMMAND_CALL_LUA_RULES which invokes gadget:RecvSkirmishAIMessage synchronously.
 *
 * AI options (from [AIOPTIONS] in the start script):
 *   port              TCP port Python is listening on
 *   decision_interval Frames between decisions (default 32)
 *   attack_snap_radius Snap radius in game units for "attack" commands (default 60)
 */

/* BUILDING_AI and BUILDING_SKIRMISH_AI are passed via -D compiler flags */
#ifndef BUILDING_AI
#  define BUILDING_AI
#endif
#ifndef BUILDING_SKIRMISH_AI
#  define BUILDING_SKIRMISH_AI
#endif

#define _POSIX_C_SOURCE 200809L

#include "ExternalAI/Interface/aidefines.h"
#include "ExternalAI/Interface/AISEvents.h"
#include "ExternalAI/Interface/AISCommands.h"
#include "ExternalAI/Interface/SSkirmishAICallback.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <math.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

/* Spring CMD constants (from Sim/Unit/CommandAI/Command.h) */
#define CMD_STOP   0
#define CMD_ATTACK 2
#define CMD_MOVE   10
#define CMD_FIGHT  16

#define SQUARE_SIZE       8
#define FIRST_FRAME       30   /* wait for game setup before contacting Python */
#define MAX_UNITS         512
#define OBS_BUF           131072
#define LINE_BUF          65536

/* ---- Global state -------------------------------------------------------- */

static const struct SSkirmishAICallback *cb = NULL;
static int ai_id = -1;

/* TCP */
static int sock_fd = -1;
/* Simple line reader state */
static char recv_buf[LINE_BUF];
static int  recv_start = 0;
static int  recv_end   = 0;

/* Config from AI options */
static int port              = 0;
static int decision_interval = 32;
static float snap_radius     = 60.0f;
static int terrain_w         = 64;
static int terrain_h         = 64;

/* Map */
static float map_x = 1.0f;  /* in game units (elmos) */
static float map_z = 1.0f;

/* Number of BAR armor types, queried from the engine in EVENT_INIT (STATIC API). */
static int num_armor_types = 14;

/* Episode */
typedef enum { S_INIT, S_RUNNING } BridgeState;
static BridgeState bridge_state = S_INIT;
static int episode             = -1;
static int episode_start_frame = 0;
static int prev_decision_frame = -1;
static int next_decision_frame = -1;

/* Unit IDs returned by the gadget after reset (comma-separated) */
static char spawned_units[4096] = "";

/* Observation write buffer */
static char obs[OBS_BUF];
static int  obs_len;

/* Receive line buffer */
static char line[LINE_BUF];

/* ---- Logging ------------------------------------------------------------- */

static void log_msg(const char *msg) {
    cb->Log_log(ai_id, msg);
}

/* ---- JSON write helpers -------------------------------------------------- */

static void obs_reset(void) { obs_len = 0; }

static void obs_str(const char *s) {
    int n = (int)strlen(s);
    if (obs_len + n < OBS_BUF - 1) {
        memcpy(obs + obs_len, s, n);
        obs_len += n;
    }
}

static void obs_printf(const char *fmt, ...) {
    char tmp[256];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(tmp, sizeof(tmp), fmt, ap);
    va_end(ap);
    obs_str(tmp);
}

/* ---- JSON read helpers --------------------------------------------------- */

/* Return pointer to value after "key": in JSON, or NULL. */
static const char *jfind(const char *json, const char *key) {
    char search[128];
    snprintf(search, sizeof(search), "\"%s\":", key);
    return strstr(json, search);
}

static int jstr(const char *json, const char *key, char *out, int maxlen) {
    char search[128];
    snprintf(search, sizeof(search), "\"%s\":\"", key);
    const char *p = strstr(json, search);
    if (!p) return 0;
    p += strlen(search);
    int i = 0;
    while (*p && *p != '"' && i < maxlen - 1) out[i++] = *p++;
    out[i] = '\0';
    return 1;
}

static int jint(const char *json, const char *key, int def) {
    const char *p = jfind(json, key);
    if (!p) return def;
    p = strchr(p, ':');
    if (!p) return def;
    return atoi(p + 1);
}

/* Advance past one JSON object { ... } starting at *p; return pointer after }. */
static const char *skip_object(const char *p) {
    int depth = 0;
    while (*p) {
        if (*p == '{') depth++;
        else if (*p == '}') { depth--; if (depth == 0) return p + 1; }
        else if (*p == '"') { p++; while (*p && *p != '"') { if (*p == '\\') p++; p++; } }
        p++;
    }
    return p;
}

/* Find start of array value for key, return pointer just after '['. */
static const char *jarray_start(const char *json, const char *key) {
    char search[128];
    snprintf(search, sizeof(search), "\"%s\":[", key);
    const char *p = strstr(json, search);
    if (!p) return NULL;
    return p + strlen(search);
}

/* ---- TCP ----------------------------------------------------------------- */

static int tcp_connect(int p) {
    sock_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (sock_fd < 0) return 0;
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port   = htons((uint16_t)p);
    inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);
    if (connect(sock_fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        close(sock_fd); sock_fd = -1; return 0;
    }
    recv_start = recv_end = 0;
    return 1;
}

static void tcp_send(void) {
    obs[obs_len++] = '\n';
    obs[obs_len] = '\0';
    send(sock_fd, obs, obs_len, 0);
}

/* Read one newline-terminated line into line[]. Returns 1 on success. */
static int tcp_recv(void) {
    int out = 0;
    while (1) {
        /* Scan buffered data for newline */
        for (int i = recv_start; i < recv_end; i++) {
            if (recv_buf[i] == '\n') {
                int len = i - recv_start;
                if (len > LINE_BUF - 1) len = LINE_BUF - 1;
                memcpy(line, recv_buf + recv_start, len);
                line[len] = '\0';
                recv_start = i + 1;
                return 1;
            }
        }
        /* No newline yet — compact and read more */
        int remaining = recv_end - recv_start;
        if (remaining > 0)
            memmove(recv_buf, recv_buf + recv_start, remaining);
        recv_start = 0;
        recv_end   = remaining;
        int n = (int)recv(sock_fd, recv_buf + recv_end,
                          sizeof(recv_buf) - recv_end - 1, 0);
        if (n <= 0) return 0;
        recv_end += n;
        recv_buf[recv_end] = '\0';
        (void)out;
    }
}

/* ---- Unit helpers -------------------------------------------------------- */

static const char *command_name(int cmd_id) {
    switch (cmd_id) {
        case CMD_MOVE:   return "move";
        case CMD_FIGHT:  return "fight";
        case CMD_ATTACK: return "attack";
        case CMD_STOP:   return "stop";
        default:         return "other";
    }
}

/* Append one unit's JSON object to obs[]. */
static void append_unit(int uid, int team_id) {
    float pos[3] = {0};
    float vel[3] = {0};
    cb->Unit_getPos(ai_id, uid, pos);
    cb->Unit_getVel(ai_id, uid, vel);
    float health  = cb->Unit_getHealth(ai_id, uid);
    int   def_id  = cb->Unit_getDef(ai_id, uid);
    float max_hp  = (def_id >= 0) ? cb->UnitDef_getHealth(ai_id, def_id) : 0.0f;
    const char *uname = (def_id >= 0) ? cb->UnitDef_getName(ai_id, def_id) : "unknown";

    /* Current command */
    int n_cmds = cb->Unit_getCurrentCommands(ai_id, uid);
    const char *cmd_str = "none";
    float tgt_x = 0.0f, tgt_z = 0.0f;
    int has_target = 0;
    if (n_cmds > 0) {
        int cmd_id = cb->Unit_CurrentCommand_getId(ai_id, uid, 0);
        cmd_str = command_name(cmd_id);
        float params[10];
        int np = cb->Unit_CurrentCommand_getParams(ai_id, uid, 0, params, 10);
        if (np == 1) {
            /* Unit target (e.g. attack) */
            int tgt_uid = (int)params[0];
            float tpos[3] = {0};
            cb->Unit_getPos(ai_id, tgt_uid, tpos);
            tgt_x = tpos[0] / map_x;
            tgt_z = tpos[2] / map_z;
            has_target = 1;
        } else if (np >= 3) {
            tgt_x = params[0] / map_x;
            tgt_z = params[2] / map_z;
            has_target = 1;
        }
    }

    obs_printf(
        "{\"id\":%d,\"team\":%d,\"unit\":\"%s\","
        "\"x\":%.6f,\"z\":%.6f,"
        "\"velocity_x\":%.6f,\"velocity_z\":%.6f,"
        "\"health\":%.2f,\"maximum_health\":%.2f,"
        "\"command\":\"%s\",",
        uid, team_id, uname,
        pos[0] / map_x, pos[2] / map_z,
        vel[0],         vel[2],
        health, max_hp,
        cmd_str
    );
    if (has_target)
        obs_printf("\"target_x\":%.6f,\"target_z\":%.6f}", tgt_x, tgt_z);
    else
        obs_str("\"target_x\":null,\"target_z\":null}");
}

/* Append unit_types dict entry for a given def_id / name, including per-weapon data. */
static void append_unit_type(int def_id, const char *name, int *first) {
    if (def_id < 0) return;
    float speed    = cb->UnitDef_getSpeed(ai_id, def_id);
    float wr       = cb->UnitDef_getMaxWeaponRange(ai_id, def_id);
    float sight    = cb->UnitDef_getLosRadius(ai_id, def_id);
    float max_hp   = cb->UnitDef_getHealth(ai_id, def_id);
    int   armor_t  = cb->UnitDef_getArmorType(ai_id, def_id);
    int   can_move = cb->UnitDef_isMoveDataAvailable(ai_id, def_id) ? 1 : 0;
    if (!(*first)) obs_str(",");
    obs_printf(
        "\"%s\":{\"speed\":%.4f,\"weapon_range\":%.4f,\"sight\":%.4f,"
        "\"maximum_health\":%.2f,\"armor_type\":%d,\"can_move\":%d,\"weapons\":[",
        name, speed, wr, sight, max_hp, armor_t, can_move);

    int n_mounts = cb->UnitDef_getWeaponMounts(ai_id, def_id);
    if (n_mounts < 0) n_mounts = 0;
    int w_first = 1;
    int n_at = (num_armor_types > 32) ? 32 : num_armor_types;
    for (int m = 0; m < n_mounts; m++) {
        int wdef_id = cb->UnitDef_WeaponMount_getWeaponDef(ai_id, def_id, m);
        if (wdef_id < 0) continue;
        float wrange  = cb->WeaponDef_getRange(ai_id, wdef_id);
        float wreload = cb->WeaponDef_getReload(ai_id, wdef_id);
        float waoe    = cb->WeaponDef_getAreaOfEffect(ai_id, wdef_id);
        int   wpar    = cb->WeaponDef_isParalyzer(ai_id, wdef_id) ? 1 : 0;
        float dmg[32] = {0.0f};
        int   n_got   = cb->WeaponDef_Damage_getTypes(ai_id, wdef_id, dmg, n_at);
        if (n_got < 0) n_got = 0;
        if (!w_first) obs_str(",");
        obs_printf("{\"range\":%.2f,\"reload\":%.4f,\"aoe\":%.2f,\"paralyzed\":%d,"
                   "\"damage_per_type\":[", wrange, wreload, waoe, wpar);
        for (int t = 0; t < n_at; t++) {
            float d = (t < n_got) ? dmg[t] : 0.0f;
            if (t > 0) obs_str(",");
            obs_printf("%.2f", d);
        }
        obs_str("]}");
        w_first = 0;
    }
    obs_str("]}");
    *first = 0;
}

/* ---- Terrain ------------------------------------------------------------- */

/* Sample the heightmap at terrain_w x terrain_h evenly spaced points and send
 * a {"type":"terrain","width":W,"height":H,"data":[h,s,w,...]} message.
 * Channels per cell: normalised height (raw/1000), slope squashed to [0,1),
 * water mask (1.0 if below sea level).  Row-major: j (north→south) outer. */
static void send_terrain(void) {
    float cell_x = map_x / terrain_w;
    float cell_z = map_z / terrain_h;

    float *heights = (float *)malloc(terrain_w * terrain_h * sizeof(float));
    if (!heights) return;

    for (int j = 0; j < terrain_h; j++) {
        for (int i = 0; i < terrain_w; i++) {
            heights[j * terrain_w + i] =
                cb->Map_getElevationAt(ai_id,
                                       (i + 0.5f) * cell_x,
                                       (j + 0.5f) * cell_z);
        }
    }

    /* generous upper bound: "%.4f,%.4f,%.1f," = ~20 chars per cell */
    int buf_size = terrain_w * terrain_h * 24 + 256;
    char *buf = (char *)malloc(buf_size);
    if (!buf) { free(heights); return; }

    int pos = snprintf(buf, buf_size,
        "{\"type\":\"terrain\",\"width\":%d,\"height\":%d,\"data\":[",
        terrain_w, terrain_h);

    for (int j = 0; j < terrain_h; j++) {
        for (int i = 0; i < terrain_w; i++) {
            float h = heights[j * terrain_w + i];

            /* Normalised height: sea level = 0, typical range [-1, ~2] */
            float h_norm   = h / 1000.0f;
            float is_water = (h < 0.0f) ? 1.0f : 0.0f;

            /* Slope via central differences (height per elmo, dimensionless) */
            float hl = (i > 0)            ? heights[j * terrain_w + (i - 1)] : h;
            float hr = (i < terrain_w - 1) ? heights[j * terrain_w + (i + 1)] : h;
            float hu = (j > 0)            ? heights[(j - 1) * terrain_w + i] : h;
            float hd = (j < terrain_h - 1) ? heights[(j + 1) * terrain_w + i] : h;
            int   di = (i > 0 && i < terrain_w - 1) ? 2 : 1;
            int   dj = (j > 0 && j < terrain_h - 1) ? 2 : 1;
            float dh_x = (hr - hl) / ((float)di * cell_x);
            float dh_z = (hd - hu) / ((float)dj * cell_z);
            float slope = sqrtf(dh_x * dh_x + dh_z * dh_z);
            float slope_norm = slope / (1.0f + slope);  /* squash to [0, 1) */

            int last = (j == terrain_h - 1 && i == terrain_w - 1);
            pos += snprintf(buf + pos, buf_size - pos,
                "%.4f,%.4f,%.1f%s",
                h_norm, slope_norm, is_water, last ? "" : ",");
        }
    }
    pos += snprintf(buf + pos, buf_size - pos, "]}\n");
    send(sock_fd, buf, pos, 0);
    free(buf);
    free(heights);
}

/* ---- Observation --------------------------------------------------------- */

static void send_observation(int frame, int frames_since_prev) {
    /* Collect all units */
    int own_ids[MAX_UNITS],   n_own   = 0;
    int enemy_ids[MAX_UNITS], n_enemy = 0;
    n_own   = cb->getTeamUnits(ai_id, own_ids,   MAX_UNITS);
    n_enemy = cb->getEnemyUnits(ai_id, enemy_ids, MAX_UNITS);
    if (n_own   < 0) n_own   = 0;
    if (n_enemy < 0) n_enemy = 0;

    obs_reset();
    obs_printf(
        "{\"type\":\"observation\",\"episode\":%d,\"frame\":%d,"
        "\"episode_frame\":%d,\"frames_since_previous_decision\":%d,"
        "\"spawned_units\":\"%s\",\"late_actions\":0,\"maximum_lead\":0,"
        "\"units\":[",
        episode, frame,
        frame - episode_start_frame, frames_since_prev,
        spawned_units
    );
    spawned_units[0] = '\0'; /* clear after first observation */

    /* Own units first, then enemies */
    for (int i = 0; i < n_own; i++) {
        if (i > 0) obs_str(",");
        int team_id = cb->Unit_getTeam(ai_id, own_ids[i]);
        append_unit(own_ids[i], team_id);
    }
    for (int i = 0; i < n_enemy; i++) {
        if (n_own > 0 || i > 0) obs_str(",");
        int team_id = cb->Unit_getTeam(ai_id, enemy_ids[i]);
        append_unit(enemy_ids[i], team_id);
    }
    obs_str("],\"unit_types\":{");

    /* Collect unique unit defs */
    int seen_defs[64];
    const char *seen_names[64];
    int n_seen = 0;
    int first = 1;
    for (int pass = 0; pass < 2; pass++) {
        int *ids = (pass == 0) ? own_ids : enemy_ids;
        int  cnt = (pass == 0) ? n_own  : n_enemy;
        for (int i = 0; i < cnt; i++) {
            int def_id = cb->Unit_getDef(ai_id, ids[i]);
            if (def_id < 0) continue;
            const char *name = cb->UnitDef_getName(ai_id, def_id);
            int already = 0;
            for (int j = 0; j < n_seen; j++) {
                if (seen_defs[j] == def_id) { already = 1; break; }
            }
            if (!already && n_seen < 64) {
                seen_defs[n_seen]  = def_id;
                seen_names[n_seen] = name;
                n_seen++;
                append_unit_type(def_id, name, &first);
            }
        }
    }
    obs_str("}}");
    tcp_send();
}

/* ---- Actions ------------------------------------------------------------- */

static void apply_actions(void) {
    const char *arr = jarray_start(line, "actions");
    if (!arr) return;
    const char *p = arr;
    while (*p && *p != ']') {
        /* Skip to next object */
        while (*p && *p != '{' && *p != ']') p++;
        if (*p != '{') break;
        const char *obj_start = p;
        const char *obj_end   = skip_object(p);
        /* Temporarily null-terminate the object for parsing */
        char obj[512];
        int obj_len = (int)(obj_end - obj_start);
        if (obj_len >= (int)sizeof(obj)) { p = obj_end; continue; }
        memcpy(obj, obj_start, obj_len);
        obj[obj_len] = '\0';

        int uid = jint(obj, "unit", -1);
        char cmd[32] = "continue";
        jstr(obj, "command", cmd, sizeof(cmd));
        float tx = 0.0f, tz = 0.0f;
        const char *px = jfind(obj, "x");
        const char *pz = jfind(obj, "z");
        if (px) tx = (float)atof(strchr(px, ':') + 1);
        if (pz) tz = (float)atof(strchr(pz, ':') + 1);
        if (tx < 0.0f) tx = 0.0f; if (tx > 1.0f) tx = 1.0f;
        if (tz < 0.0f) tz = 0.0f; if (tz > 1.0f) tz = 1.0f;

        float wx = tx * map_x;
        float wz = tz * map_z;
        float ground_y = cb->Map_getElevationAt(ai_id, wx, wz);
        float wpos[3] = { wx, ground_y, wz };

        if (uid >= 0 && strcmp(cmd, "continue") != 0) {
            if (strcmp(cmd, "stop") == 0) {
                struct SStopUnitCommand scmd = { uid, -1, 0, INT_MAX };
                cb->Engine_handleCommand(ai_id, COMMAND_TO_ID_ENGINE, -1,
                                         COMMAND_UNIT_STOP, &scmd);
            } else if (strcmp(cmd, "move") == 0) {
                struct SMoveUnitCommand scmd = { uid, -1, 0, INT_MAX, wpos };
                cb->Engine_handleCommand(ai_id, COMMAND_TO_ID_ENGINE, -1,
                                         COMMAND_UNIT_MOVE, &scmd);
            } else if (strcmp(cmd, "fight") == 0) {
                struct SFightUnitCommand scmd = { uid, -1, 0, INT_MAX, wpos };
                cb->Engine_handleCommand(ai_id, COMMAND_TO_ID_ENGINE, -1,
                                         COMMAND_UNIT_FIGHT, &scmd);
            } else if (strcmp(cmd, "attack") == 0) {
                /* Snap to nearest enemy within snap_radius */
                int nearby[MAX_UNITS];
                int n = cb->getEnemyUnitsIn(ai_id, wpos, snap_radius, 0, nearby, MAX_UNITS);
                if (n > 0) {
                    int best = -1;
                    float best_d = 1e30f;
                    for (int i = 0; i < n; i++) {
                        float ep[3] = {0};
                        cb->Unit_getPos(ai_id, nearby[i], ep);
                        float dx = ep[0] - wx, dz = ep[2] - wz;
                        float d  = dx * dx + dz * dz;
                        if (d < best_d) { best_d = d; best = nearby[i]; }
                    }
                    if (best >= 0) {
                        struct SAttackUnitCommand scmd = { uid, -1, 0, INT_MAX, best };
                        cb->Engine_handleCommand(ai_id, COMMAND_TO_ID_ENGINE, -1,
                                                 COMMAND_UNIT_ATTACK, &scmd);
                    }
                }
            }
        }
        p = obj_end;
    }
}

/* ---- Reset --------------------------------------------------------------- */

/* Call the synced gadget to reset the scenario. The gadget destroys/spawns
 * units and returns the spawned unit IDs as a comma-separated string. */
static void call_gadget_reset(const char *reset_json) {
    /* Build the message: "rl_reset|episode|team,unit,x,z;...|fire0,move0,fire1,move1" */
    char msg[4096];
    int ep = jint(reset_json, "episode", 0);
    episode = ep;

    /* States */
    const char *states_arr = jarray_start(reset_json, "states");
    int s[4] = {2, 1, 0, 0};
    if (states_arr) sscanf(states_arr, "%d,%d,%d,%d", &s[0], &s[1], &s[2], &s[3]);

    /* Spawns */
    char spawns_str[2048] = "";
    const char *arr = jarray_start(reset_json, "spawns");
    if (arr) {
        const char *p = arr;
        int first = 1;
        while (*p && *p != ']') {
            while (*p && *p != '{' && *p != ']') p++;
            if (*p != '{') break;
            const char *obj_end = skip_object(p);
            char obj[256];
            int olen = (int)(obj_end - p);
            if (olen < (int)sizeof(obj)) {
                memcpy(obj, p, olen); obj[olen] = '\0';
                int team = jint(obj, "team", 0);
                char unit[64]; jstr(obj, "unit", unit, sizeof(unit));
                float sx = 0.5f, sz = 0.5f;
                const char *px = jfind(obj, "x"), *pz = jfind(obj, "z");
                if (px) sx = (float)atof(strchr(px, ':') + 1);
                if (pz) sz = (float)atof(strchr(pz, ':') + 1);
                char entry[128];
                snprintf(entry, sizeof(entry), "%s%d,%s,%.6f,%.6f", first ? "" : ";",
                         team, unit, sx, sz);
                strncat(spawns_str, entry, sizeof(spawns_str) - strlen(spawns_str) - 1);
                first = 0;
            }
            p = obj_end;
        }
    }

    snprintf(msg, sizeof(msg), "rl_reset|%d|%s|%d,%d,%d,%d",
             ep, spawns_str, s[0], s[1], s[2], s[3]);

    char response[MAX_RESPONSE_SIZE];
    memset(response, 0, sizeof(response));
    struct SCallLuaRulesCommand lcmd = { msg, -1, response };
    cb->Engine_handleCommand(ai_id, COMMAND_TO_ID_ENGINE, -1,
                             COMMAND_CALL_LUA_RULES, &lcmd);

    /* Response is the comma-separated list of spawned unit IDs */
    strncpy(spawned_units, response, sizeof(spawned_units) - 1);
    spawned_units[sizeof(spawned_units) - 1] = '\0';
}

/* ---- Main event handler -------------------------------------------------- */

EXPORT(int) handleEvent(int skirmishAIId, int topic, const void *data) {
    switch (topic) {

    case EVENT_INIT: {
        ai_id = skirmishAIId;
        const struct SInitEvent *e = (const struct SInitEvent *)data;
        cb = e->callback;

        /* Read AI options */
        const char *port_str = cb->SkirmishAI_OptionValues_getValueByKey(ai_id, "port");
        if (port_str) port = atoi(port_str);
        const char *di_str  = cb->SkirmishAI_OptionValues_getValueByKey(ai_id, "decision_interval");
        if (di_str)  decision_interval = atoi(di_str);
        const char *sr_str  = cb->SkirmishAI_OptionValues_getValueByKey(ai_id, "attack_snap_radius");
        if (sr_str)  snap_radius = (float)atof(sr_str);
        const char *tw_str  = cb->SkirmishAI_OptionValues_getValueByKey(ai_id, "terrain_w");
        if (tw_str)  terrain_w = atoi(tw_str);
        const char *th_str  = cb->SkirmishAI_OptionValues_getValueByKey(ai_id, "terrain_h");
        if (th_str)  terrain_h = atoi(th_str);

        /* Map size in game units */
        map_x = (float)(cb->Map_getWidth(ai_id)  * SQUARE_SIZE);
        map_z = (float)(cb->Map_getHeight(ai_id) * SQUARE_SIZE);

        /* Armor type count is a static global property of the game. */
        int n_armor = cb->WeaponDef_getNumDamageTypes(ai_id);
        if (n_armor > 0) num_armor_types = n_armor;

        if (!tcp_connect(port)) {
            log_msg("[BridgeAI] cannot connect to Python");
            return -1;
        }
        log_msg("[BridgeAI] connected to Python");
        break;
    }

    case EVENT_UPDATE: {
        const struct SUpdateEvent *e = (const struct SUpdateEvent *)data;
        int frame = e->frame;

        if (bridge_state == S_INIT) {
            if (frame < FIRST_FRAME) break;

            /* First contact: announce map size + armor type count, then terrain, then reset */
            obs_reset();
            obs_printf("{\"type\":\"ready\",\"frame\":%d,\"num_armor_types\":%d,\"map\":"
                       "{\"size_x\":%.1f,\"size_z\":%.1f,\"name\":\"\"}}", /* name unused */
                       frame, num_armor_types, map_x, map_z);
            tcp_send();
            send_terrain();

            if (!tcp_recv()) { log_msg("[BridgeAI] connection lost"); return -1; }
            char type[32]; jstr(line, "type", type, sizeof(type));
            if (strcmp(type, "quit") == 0) { log_msg("[BridgeAI] quit"); return -1; }
            if (strcmp(type, "reset") != 0) break;

            call_gadget_reset(line);
            episode_start_frame = frame;
            prev_decision_frame = frame;
            next_decision_frame = frame + decision_interval;
            send_observation(frame, 0);
            bridge_state = S_RUNNING;

            if (!tcp_recv()) { log_msg("[BridgeAI] connection lost"); return -1; }
            jstr(line, "type", type, sizeof(type));
            if (strcmp(type, "actions") == 0) apply_actions();
        }
        else if (bridge_state == S_RUNNING) {
            if (frame < next_decision_frame) break;

            int frames_since = frame - prev_decision_frame;
            prev_decision_frame = frame;
            next_decision_frame = frame + decision_interval;

            send_observation(frame, frames_since);
            if (!tcp_recv()) { log_msg("[BridgeAI] connection lost"); return -1; }

            char type[32]; jstr(line, "type", type, sizeof(type));
            if (strcmp(type, "quit") == 0) { log_msg("[BridgeAI] quit"); return -1; }
            if (strcmp(type, "actions") == 0) {
                apply_actions();
            } else if (strcmp(type, "reset") == 0) {
                call_gadget_reset(line);
                episode_start_frame = frame;
                prev_decision_frame = frame;
                next_decision_frame = frame + decision_interval;
                send_observation(frame, 0);
                if (!tcp_recv()) { log_msg("[BridgeAI] connection lost"); return -1; }
                jstr(line, "type", type, sizeof(type));
                if (strcmp(type, "actions") == 0) apply_actions();
            }
        }
        break;
    }

    case EVENT_RELEASE: {
        if (sock_fd >= 0) { close(sock_fd); sock_fd = -1; }
        break;
    }

    default:
        break;
    }
    return 0;
}
