const { createApp, ref, onMounted, nextTick } = Vue;

createApp({
    setup() {
        // State
        const currentView = ref('dashboard'); // 'dashboard' | 'graph'
        const workflows = ref([]);
        const activeWorkflowId = ref(null);
        const jsonInput = ref('');
        const submitStatus = ref('');
        const submitError = ref(false);
        const isSubmitting = ref(false);
        const selectedTask = ref(null);
        
        let ws = null;
        let graphLib = null; // Store dagre-d3 instance data

        // Methods
        const fetchWorkflows = async () => {
            try {
                const res = await fetch('/api/workflows');
                if (res.ok) {
                    workflows.value = await res.json();
                }
            } catch (e) {
                console.error("Failed to fetch workflows", e);
            }
        };

        const goHome = () => {
            currentView.value = 'dashboard';
            activeWorkflowId.value = null;
            selectedTask.value = null;
            if (ws) ws.close();
            fetchWorkflows();
        };

        const handleFile = async (e) => {
            const file = e.target.files?.[0];
            if (!file) return;
            jsonInput.value = await file.text();
        };

        const submitWorkflow = async () => {
            submitStatus.value = '';
            submitError.value = false;
            let payload;

            try {
                payload = JSON.parse(jsonInput.value);
            } catch (e) {
                submitStatus.value = "Invalid JSON syntax.";
                submitError.value = true;
                return;
            }

            // Ensure ID exists or basic validation
            const wfId = payload?.workflow?.id;
            if (!wfId) {
                submitStatus.value = "JSON must include { 'workflow': { 'id': ... } }";
                submitError.value = true;
                return;
            }

            isSubmitting.value = true;
            try {
                const res = await fetch("/workflows", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(payload),
                });

                if (!res.ok) {
                    const txt = await res.text();
                    throw new Error(txt);
                }

                submitStatus.value = "Success!";
                setTimeout(() => viewGraph(wfId), 500);
            } catch (e) {
                submitStatus.value = "Submit failed: " + e.message;
                submitError.value = true;
            } finally {
                isSubmitting.value = false;
            }
        };

        const viewGraph = async (id) => {
            activeWorkflowId.value = id;
            currentView.value = 'graph';
            
            // Wait for DOM to update so SVG exists
            await nextTick();
            
            try {
                const res = await fetch(`/api/workflows/${encodeURIComponent(id)}/graph`);
                if (!res.ok) throw new Error("Failed to load graph");
                const data = await res.json();
                
                initGraph(data);
                connectWS(id);
            } catch (e) {
                alert("Error loading graph: " + e.message);
                goHome();
            }
        };

        // Date formatting helpers
        const formatDate = (str) => {
            if (!str) return '';
            return new Date(str).toLocaleString();
        };
        const formatTime = (str) => {
            if (!str) return '';
            return new Date(str).toLocaleTimeString();
        };

        const getStatusColor = (s) => {
            switch(s) {
                case 'completed': return 'bg-green-500/20 text-green-400 border border-green-500/50';
                case 'failed': return 'bg-red-500/20 text-red-400 border border-red-500/50';
                case 'running': return 'bg-blue-500/20 text-blue-400 border border-blue-500/50';
                case 'runnable': return 'bg-slate-600/20 text-slate-300 border border-slate-500/50';
                default: return 'bg-slate-800 text-slate-500';
            }
        };

        // --- GRAPH LOGIC ---
        const initGraph = (data) => {
            // Create Dagre Graph
            const g = new dagreD3.graphlib.Graph().setGraph({
                rankdir: "LR", // Left to Right
                nodesep: 50,
                ranksep: 100,
                marginx: 20,
                marginy: 20
            });

            const tasksById = new Map();
            data.tasks.forEach(t => tasksById.set(t.id, t));

            const wfLabel = (data.workflow_name || data.workflow_id || 'WORKFLOW');
            g.setNode("ROOT", { 
              label: wfLabel,
              class: "workflow",
              shape: "circle",
              padding: 18
            });

            data.tasks.forEach(t => {
                g.setNode(t.id, {
                    label: `${t.id}`, 
                    class: (t.status || 'pending').toLowerCase(),
                    padding: 15,
                    rx: 5, ry: 5 // rounded corners
                });
            });

            // 3. Add Edges (Dependencies)
            const hasIncoming = new Set();
            data.edges.forEach(e => {
                g.setEdge(e.from, e.to, {
                    arrowhead: "vee",
                    lineInterpolate: "basis",
                    style: "stroke-width: 2px;"
                });
                hasIncoming.add(e.to);
            });

            // 4. Connect ROOT to tasks with no dependencies
            data.tasks.forEach(t => {
                if (!hasIncoming.has(t.id)) {
                    g.setEdge("ROOT", t.id, {
                        arrowhead: "vee",
                        style: "stroke-dasharray: 5, 5; stroke-opacity: 0.5;"
                    });
                }
            });

            // Render
            const svg = d3.select("#svg-canvas");
            svg.selectAll("*").remove(); // clear previous
            const inner = svg.append("g");
            
            const render = new dagreD3.render();
            render(inner, g);

            // Zooming
            const zoom = d3.zoom().on("zoom", (event) => {
                inner.attr("transform", event.transform);
            });
            svg.call(zoom);

            // Initial Center
            const svgNode = svg.node();
            const { width, height } = svgNode.getBoundingClientRect();
            const graphHeight = g.graph().height;
            const graphWidth = g.graph().width;
            
            const scale = Math.min(width / (graphWidth + 100), height / (graphHeight + 100), 1);
            const xCenterOffset = (width - graphWidth * scale) / 2;
            const yCenterOffset = (height - graphHeight * scale) / 2;
            
            svg.call(zoom.transform, d3.zoomIdentity.translate(20, height/2 - (graphHeight*scale)/2).scale(scale));

            // Click Interaction
            inner.selectAll("g.node").on("click", (e, id) => {
                if (id === 'ROOT') return;
                selectedTask.value = tasksById.get(id);
            });

            // Store for updates
            graphLib = { g, render, inner, tasksById };
        };

        const connectWS = (id) => {
            if (ws) ws.close();
            const proto = location.protocol === "https:" ? "wss" : "ws";
            ws = new WebSocket(`${proto}://${location.host}/ws`);

            ws.onopen = () => {
                ws.send(JSON.stringify({ type: "subscribe", workflow_id: id }));
            };

            ws.onmessage = (ev) => {
                const msg = JSON.parse(ev.data);
                if (msg.type === "task.updated") {
                    updateGraphNode(msg.payload);
                }
            };
        };

        const updateGraphNode = (update) => {
            if (!graphLib) return;
            const { g, render, inner, tasksById } = graphLib;
            
            // Sync Data
            const t = tasksById.get(update.task_id);
            if (!t) return;
            Object.assign(t, update); // Merge updates

            // Sync Visuals
            const node = g.node(update.task_id);
            if (node) {
                node.class = (update.status || 'pending').toLowerCase();
                // Rerender efficiently
                render(inner, g);
                
                // If this is the currently selected task, force reactivity update
                if (selectedTask.value && selectedTask.value.id === update.task_id) {
                    selectedTask.value = { ...t }; 
                }
            }
        };

        // Lifecycle
        onMounted(() => {
            fetchWorkflows();
        });

        return {
            currentView,
            workflows,
            activeWorkflowId,
            jsonInput,
            submitStatus,
            submitError,
            isSubmitting,
            selectedTask,
            // Actions
            handleFile,
            submitWorkflow,
            viewGraph,
            goHome,
            formatDate,
            formatTime,
            getStatusColor
        };
    }
}).mount('#app');