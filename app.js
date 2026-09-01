document.addEventListener('DOMContentLoaded', () => {
    // UI Screen Sections
    const screens = {
        'screen-hero': document.getElementById('screen-hero'),
        'screen-dashboard': document.getElementById('screen-dashboard'),
        'screen-workspace': document.getElementById('screen-workspace'),
        'screen-quote': document.getElementById('screen-quote'),
        'screen-success': document.getElementById('screen-success')
    };

    // Active States
    let currentScreen = 'screen-hero';
    let activeRoom = null;
    let activeTile = null;
    
    // Canvas & Contexts
    const canvas = document.getElementById('visualizer-canvas');
    const ctx = canvas.getContext('2d');
    
    // Offscreen Canvas for advanced loop tiling & compositing
    const offscreenCanvas = document.createElement('canvas');
    const offCtx = offscreenCanvas.getContext('2d');
    
    // Compare Canvases (Before / After Split view)
    const compareCanvasBefore = document.getElementById('compare-canvas-before');
    const compareCtxBefore = compareCanvasBefore.getContext('2d');
    const compareCanvasAfter = document.getElementById('compare-canvas-after');
    const compareCtxAfter = compareCanvasAfter.getContext('2d');
    const compareAfterWrapper = document.getElementById('compare-after-wrapper');
    const splitSliderBar = document.getElementById('split-slider-bar');
    
    // SVG Floor Outlines overlay
    const outlineSvg = document.getElementById('outline-svg');
    
    // Image Loader Cache
    const imageCache = {
        roomImage: new Image(),
        maskImage: new Image(),
        tileImage: new Image(),
        wallMaskImage: new Image(),
        wallTileImage: new Image()
    };
    
    // Tiling Engine parameters & state
    let visualizerState = {
        target: 'floor', // 'floor' or 'wall'
        // Floor parameters
        scale: 0.18,
        rotation: 0,
        brightness: 1.0,
        pattern: 'grid',
        groutWidth: 0,
        groutColor: '#cccccc',
        finish: 'matte',
        slab: 0,
        shadowStrength: 0.55,
        tilesX: 6,
        tilesY: 5,
        maskDataUrl: null,
        polygon: [],
        floorQuad: [],
        
        // Wall parameters
        wallScale: 0.18,
        wallRotation: 0,
        wallBrightness: 1.0,
        wallPattern: 'grid',
        wallGroutWidth: 0,
        wallGroutColor: '#cccccc',
        wallFinish: 'matte',
        wallSlab: 0,
        wallShadowStrength: 0.55,
        wallTilesX: 5,
        wallTilesY: 7,
        wallMaskDataUrl: null,
        wallPolygon: [],
        wallQuads: [],
        
        mode: 'design', // 'design' or 'compare'
        activeFloorTileId: 'tile-1',
        activeWallTileId: null,
        zoom: 1.0,
        
        // Server-rendered composited image (Option B)
        serverRenderedImage: null,
        
        // Detected area info
        floorArea: null,
        wallArea: null,
        detectedObstacles: {},

        // Room perspective (from /api/segment)
        pixelsPerMeter: 0,
        perspective: null,
    };

    // Parse "… | 600 x 900 mm" -> {w:600, h:900}; falls back to 600x600.
    function parseTileMM(specs) {
        const m = (specs || '').match(/(\d{2,4})\s*[x×]\s*(\d{2,4})\s*mm/i);
        if (m) return { w: parseFloat(m[1]), h: parseFloat(m[2]) };
        return { w: 600, h: 600 };
    }
    
    // Debounce timer for server calls
    let visualizeDebounceTimer = null;
    
    // Blob URLs for cached room & tile files (for API calls)
    let cachedRoomBlob = null;
    let cachedFloorMaskBlob = null;
    let cachedWallMaskBlob = null;
    let cachedWallFgBlob = null;

    // Curated Preset Rooms
    const roomsData = [
        {
                "id": "room-1",
                "name": "Living Room 1",
                "sub": "Curated Living Room space template",
                "img": "images_templates/images_living_room/floor-tiles-ideas-for-living-room.jpg",
                "cardImg": "images_templates/images_living_room/floor-tiles-ideas-for-living-room.jpg",
                "baseScale": 1.0,
                "type": "living_room"
        },
        {
                "id": "room-2",
                "name": "Living Room 2",
                "sub": "Curated Living Room space template",
                "img": "images_templates/images_living_room/images (1).jpg",
                "cardImg": "images_templates/images_living_room/images (1).jpg",
                "baseScale": 1.0,
                "type": "living_room"
        },
        {
                "id": "room-3",
                "name": "Living Room 3",
                "sub": "Curated Living Room space template",
                "img": "images_templates/images_living_room/images (2).jpg",
                "cardImg": "images_templates/images_living_room/images (2).jpg",
                "baseScale": 1.0,
                "type": "living_room"
        },
        {
                "id": "room-4",
                "name": "Living Room 4",
                "sub": "Curated Living Room space template",
                "img": "images_templates/images_living_room/images (3).jpg",
                "cardImg": "images_templates/images_living_room/images (3).jpg",
                "baseScale": 1.0,
                "type": "living_room"
        },
        {
                "id": "room-5",
                "name": "Living Room 5",
                "sub": "Curated Living Room space template",
                "img": "images_templates/images_living_room/images (4).jpg",
                "cardImg": "images_templates/images_living_room/images (4).jpg",
                "baseScale": 1.0,
                "type": "living_room"
        },
        {
                "id": "room-6",
                "name": "Living Room 6",
                "sub": "Curated Living Room space template",
                "img": "images_templates/images_living_room/images (5).jpg",
                "cardImg": "images_templates/images_living_room/images (5).jpg",
                "baseScale": 1.0,
                "type": "living_room"
        },
        {
                "id": "room-7",
                "name": "Living Room 7",
                "sub": "Curated Living Room space template",
                "img": "images_templates/images_living_room/images.jpg",
                "cardImg": "images_templates/images_living_room/images.jpg",
                "baseScale": 1.0,
                "type": "living_room"
        },
        {
                "id": "room-8",
                "name": "Living Room 8",
                "sub": "Curated Living Room space template",
                "img": "images_templates/images_living_room/WhatsApp-Image-2026-05-05-at-7.02.40-PM.jpg",
                "cardImg": "images_templates/images_living_room/WhatsApp-Image-2026-05-05-at-7.02.40-PM.jpg",
                "baseScale": 1.0,
                "type": "living_room"
        },
        {
                "id": "room-9",
                "name": "Bedroom 1",
                "sub": "Curated Bedroom space template",
                "img": "images_templates/images_bedroom/download.jpg",
                "cardImg": "images_templates/images_bedroom/download.jpg",
                "baseScale": 1.0,
                "type": "bedroom"
        },
        {
                "id": "room-10",
                "name": "Bedroom 2",
                "sub": "Curated Bedroom space template",
                "img": "images_templates/images_bedroom/images (1).jpg",
                "cardImg": "images_templates/images_bedroom/images (1).jpg",
                "baseScale": 1.0,
                "type": "bedroom"
        },
        {
                "id": "room-11",
                "name": "Bedroom 3",
                "sub": "Curated Bedroom space template",
                "img": "images_templates/images_bedroom/images (2).jpg",
                "cardImg": "images_templates/images_bedroom/images (2).jpg",
                "baseScale": 1.0,
                "type": "bedroom"
        },
        {
                "id": "room-12",
                "name": "Bedroom 4",
                "sub": "Curated Bedroom space template",
                "img": "images_templates/images_bedroom/images (3).jpg",
                "cardImg": "images_templates/images_bedroom/images (3).jpg",
                "baseScale": 1.0,
                "type": "bedroom"
        },
        {
                "id": "room-13",
                "name": "Bedroom 5",
                "sub": "Curated Bedroom space template",
                "img": "images_templates/images_bedroom/images (4).jpg",
                "cardImg": "images_templates/images_bedroom/images (4).jpg",
                "baseScale": 1.0,
                "type": "bedroom"
        },
        {
                "id": "room-14",
                "name": "Bedroom 6",
                "sub": "Curated Bedroom space template",
                "img": "images_templates/images_bedroom/images (5).jpg",
                "cardImg": "images_templates/images_bedroom/images (5).jpg",
                "baseScale": 1.0,
                "type": "bedroom"
        },
        {
                "id": "room-15",
                "name": "Bedroom 7",
                "sub": "Curated Bedroom space template",
                "img": "images_templates/images_bedroom/images.jpg",
                "cardImg": "images_templates/images_bedroom/images.jpg",
                "baseScale": 1.0,
                "type": "bedroom"
        },
        {
                "id": "room-16",
                "name": "Bedroom 8",
                "sub": "Curated Bedroom space template",
                "img": "images_templates/images_bedroom/tiles-for-bedroom.jpg",
                "cardImg": "images_templates/images_bedroom/tiles-for-bedroom.jpg",
                "baseScale": 1.0,
                "type": "bedroom"
        },
        {
                "id": "room-17",
                "name": "Kitchen 1",
                "sub": "Curated Kitchen space template",
                "img": "images_templates/images_kitchen/images (1).jpg",
                "cardImg": "images_templates/images_kitchen/images (1).jpg",
                "baseScale": 1.0,
                "type": "kitchen"
        },
        {
                "id": "room-18",
                "name": "Kitchen 2",
                "sub": "Curated Kitchen space template",
                "img": "images_templates/images_kitchen/images (2).jpg",
                "cardImg": "images_templates/images_kitchen/images (2).jpg",
                "baseScale": 1.0,
                "type": "kitchen"
        },
        {
                "id": "room-19",
                "name": "Kitchen 3",
                "sub": "Curated Kitchen space template",
                "img": "images_templates/images_kitchen/images (3).jpg",
                "cardImg": "images_templates/images_kitchen/images (3).jpg",
                "baseScale": 1.0,
                "type": "kitchen"
        },
        {
                "id": "room-20",
                "name": "Kitchen 4",
                "sub": "Curated Kitchen space template",
                "img": "images_templates/images_kitchen/images (4).jpg",
                "cardImg": "images_templates/images_kitchen/images (4).jpg",
                "baseScale": 1.0,
                "type": "kitchen"
        },
        {
                "id": "room-21",
                "name": "Kitchen 5",
                "sub": "Curated Kitchen space template",
                "img": "images_templates/images_kitchen/images (5).jpg",
                "cardImg": "images_templates/images_kitchen/images (5).jpg",
                "baseScale": 1.0,
                "type": "kitchen"
        },
        {
                "id": "room-22",
                "name": "Kitchen 6",
                "sub": "Curated Kitchen space template",
                "img": "images_templates/images_kitchen/images (6).jpg",
                "cardImg": "images_templates/images_kitchen/images (6).jpg",
                "baseScale": 1.0,
                "type": "kitchen"
        },
        {
                "id": "room-23",
                "name": "Kitchen 7",
                "sub": "Curated Kitchen space template",
                "img": "images_templates/images_kitchen/images (7).jpg",
                "cardImg": "images_templates/images_kitchen/images (7).jpg",
                "baseScale": 1.0,
                "type": "kitchen"
        },
        {
                "id": "room-24",
                "name": "Kitchen 8",
                "sub": "Curated Kitchen space template",
                "img": "images_templates/images_kitchen/images (8).jpg",
                "cardImg": "images_templates/images_kitchen/images (8).jpg",
                "baseScale": 1.0,
                "type": "kitchen"
        },
        {
                "id": "room-25",
                "name": "Kitchen 9",
                "sub": "Curated Kitchen space template",
                "img": "images_templates/images_kitchen/images.jpg",
                "cardImg": "images_templates/images_kitchen/images.jpg",
                "baseScale": 1.0,
                "type": "kitchen"
        },
        {
                "id": "room-26",
                "name": "Bathroom 1",
                "sub": "Curated Bathroom space template",
                "img": "images_templates/images_bathroom/bathroom-floortiles-homesquare.webp",
                "cardImg": "images_templates/images_bathroom/bathroom-floortiles-homesquare.webp",
                "baseScale": 1.0,
                "type": "bathroom"
        },
        {
                "id": "room-27",
                "name": "Bathroom 2",
                "sub": "Curated Bathroom space template",
                "img": "images_templates/images_bathroom/images (1).jpg",
                "cardImg": "images_templates/images_bathroom/images (1).jpg",
                "baseScale": 1.0,
                "type": "bathroom"
        },
        {
                "id": "room-28",
                "name": "Bathroom 3",
                "sub": "Curated Bathroom space template",
                "img": "images_templates/images_bathroom/images (2).jpg",
                "cardImg": "images_templates/images_bathroom/images (2).jpg",
                "baseScale": 1.0,
                "type": "bathroom"
        },
        {
                "id": "room-29",
                "name": "Bathroom 4",
                "sub": "Curated Bathroom space template",
                "img": "images_templates/images_bathroom/images (3).jpg",
                "cardImg": "images_templates/images_bathroom/images (3).jpg",
                "baseScale": 1.0,
                "type": "bathroom"
        },
        {
                "id": "room-30",
                "name": "Bathroom 5",
                "sub": "Curated Bathroom space template",
                "img": "images_templates/images_bathroom/images (4).jpg",
                "cardImg": "images_templates/images_bathroom/images (4).jpg",
                "baseScale": 1.0,
                "type": "bathroom"
        },
        {
                "id": "room-31",
                "name": "Bathroom 6",
                "sub": "Curated Bathroom space template",
                "img": "images_templates/images_bathroom/images (5).jpg",
                "cardImg": "images_templates/images_bathroom/images (5).jpg",
                "baseScale": 1.0,
                "type": "bathroom"
        },
        {
                "id": "room-32",
                "name": "Bathroom 7",
                "sub": "Curated Bathroom space template",
                "img": "images_templates/images_bathroom/images (6).jpg",
                "cardImg": "images_templates/images_bathroom/images (6).jpg",
                "baseScale": 1.0,
                "type": "bathroom"
        },
        {
                "id": "room-33",
                "name": "Bathroom 8",
                "sub": "Curated Bathroom space template",
                "img": "images_templates/images_bathroom/images (7).jpg",
                "cardImg": "images_templates/images_bathroom/images (7).jpg",
                "baseScale": 1.0,
                "type": "bathroom"
        },
        {
                "id": "room-34",
                "name": "Bathroom 9",
                "sub": "Curated Bathroom space template",
                "img": "images_templates/images_bathroom/images (8).jpg",
                "cardImg": "images_templates/images_bathroom/images (8).jpg",
                "baseScale": 1.0,
                "type": "bathroom"
        },
        {
                "id": "room-35",
                "name": "Bathroom 10",
                "sub": "Curated Bathroom space template",
                "img": "images_templates/images_bathroom/images (9).jpg",
                "cardImg": "images_templates/images_bathroom/images (9).jpg",
                "baseScale": 1.0,
                "type": "bathroom"
        },
        {
                "id": "room-36",
                "name": "Bathroom 11",
                "sub": "Curated Bathroom space template",
                "img": "images_templates/images_bathroom/images.jpg",
                "cardImg": "images_templates/images_bathroom/images.jpg",
                "baseScale": 1.0,
                "type": "bathroom"
        }
]

    const materialsData = [
        {
            id: "tile-1",
            name: "White Statuario",
            brand: "Italian Marble",
            img: "assets/tile_1.png",
            price: 75,
            type: "marble",
            specs: "Premium Gloss Finish | 600 x 600 mm"
        },
        {
            id: "tile-2",
            name: "Grey Veined",
            brand: "Italian Marble",
            img: "assets/tile_2.png",
            price: 65,
            type: "marble",
            specs: "High Gloss Polish | 600 x 600 mm"
        },
        {
            id: "tile-3",
            name: "Classic Beige",
            brand: "Turkish Marble",
            img: "assets/tile_3.png",
            price: 62,
            type: "marble",
            specs: "Honed Semi-Gloss | 600 x 600 mm"
        },
        {
            id: "tile-4",
            name: "Oak Brown Planks",
            brand: "Natural Wood",
            img: "assets/tile_4.png",
            price: 58,
            type: "wood",
            specs: "Textured Matte | 150 x 900 mm"
        },
        {
            id: "tile-5",
            name: "Walnut Planks",
            brand: "Natural Wood",
            img: "assets/tile_5.png",
            price: 60,
            type: "wood",
            specs: "Premium Walnut Matte | 150 x 900 mm"
        },
        {
            id: "tile-6",
            name: "Light Grey",
            brand: "Terrazzo Classic",
            img: "assets/tile_6.png",
            price: 52,
            type: "ceramic",
            specs: "Satin Matte | 600 x 600 mm"
        },
        {
            id: "tile-7",
            name: "Dark Charcoal",
            brand: "Granite Stone",
            img: "assets/tile_7.png",
            price: 80,
            type: "ceramic",
            specs: "Polished Granite | 600 x 600 mm"
        }
    ];

    // Default setups
    activeRoom = roomsData[0];
    let activeFloorTile = materialsData[0];
    let activeWallTile = null;
    activeTile = activeFloorTile;

    // --- Screen Manager ---
    function showScreen(screenId) {
        Object.keys(screens).forEach(id => {
            if (screens[id]) {
                screens[id].classList.replace('active-screen', 'inactive-screen');
            }
        });
        
        const target = screens[screenId];
        if (target) {
            target.classList.replace('inactive-screen', 'active-screen');
            currentScreen = screenId;
        }

        // Highlight header navigation
        document.querySelectorAll('.nav-item').forEach(item => {
            item.classList.remove('active');
        });
        
        if (screenId === 'screen-hero') {
            document.getElementById('nav-home').classList.add('active');
        } else if (screenId === 'screen-workspace') {
            document.getElementById('nav-visualizer').classList.add('active');
            resizeCanvasToFit();
        } else if (screenId === 'screen-quote') {
            document.getElementById('nav-calculator').classList.add('active');
            updateCostEstimator();
        }

        // Highlight Dock button states
        document.querySelectorAll('.dock-btn').forEach(btn => {
            if (btn.dataset.screen === screenId) {
                btn.classList.add('active');
            } else {
                btn.classList.remove('active');
            }
        });
        
        window.scrollTo({ top: 0, behavior: 'smooth' });
    }

    // Connect dock buttons
    document.querySelectorAll('.dock-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            showScreen(btn.dataset.screen);
        });
    });

    // Theme toggler
    const themeBtn = document.getElementById('theme-btn');
    if (themeBtn) {
        themeBtn.addEventListener('click', () => {
            document.body.classList.toggle('light-theme');
            if (currentScreen === 'screen-workspace') {
                renderVisualizer();
            }
        });
    }

    // --- Collapsible Panel Mode ---
    const sidebarToggle = document.getElementById('sidebar-toggle');
    const workspaceContainer = document.getElementById('workspace-container');
    const toggleIcon = document.getElementById('sidebar-toggle-icon');
    const toggleText = document.getElementById('sidebar-toggle-text');
    
    if (sidebarToggle && workspaceContainer) {
        sidebarToggle.addEventListener('click', () => {
            const isCollapsed = workspaceContainer.classList.toggle('collapsed');
            
            if (isCollapsed) {
                toggleText.textContent = 'Show Panel';
                if (toggleIcon) toggleIcon.style.transform = 'rotate(180deg)';
            } else {
                toggleText.textContent = 'Full View';
                if (toggleIcon) toggleIcon.style.transform = 'rotate(0deg)';
            }
            
            // Wait for transition width to resolve, then resize & redraw
            setTimeout(() => {
                resizeCanvasToFit();
                renderVisualizer();
            }, 360);
        });
    }

    // --- Content Generators ---
    function initDynamicUI() {
        // 1. Populate Rooms Grid in Dashboard
        function populateRoomsDashboard(filter = 'all') {
            const dashboardPresets = document.getElementById('dashboard-presets-grid');
            if (!dashboardPresets) return;
            
            dashboardPresets.innerHTML = '';
            const filtered = filter === 'all' ? roomsData : roomsData.filter(r => r.type === filter);
            
            filtered.forEach(room => {
                const card = document.createElement('div');
                card.className = 'room-card';
                card.innerHTML = `
                    <img src="${room.cardImg}" alt="${room.name}" class="room-card-img">
                    <div class="room-card-label">${room.name}</div>
                `;
                card.addEventListener('click', () => {
                    loadRoom(room);
                    showScreen('screen-workspace');
                });
                dashboardPresets.appendChild(card);
            });
        }

        // 2. Populate Rooms Tab in Sidebar
        function populateRoomsSidebar(filter = 'all') {
            const roomsContainer = document.getElementById('rooms-list');
            if (!roomsContainer) return;
            
            roomsContainer.innerHTML = '';
            const filtered = filter === 'all' ? roomsData : roomsData.filter(r => r.type === filter);
            
            filtered.forEach(room => {
                const card = document.createElement('div');
                card.className = `room-card ${room.id === activeRoom.id ? 'active' : ''}`;
                card.dataset.id = room.id;
                card.innerHTML = `
                    <img src="${room.cardImg}" alt="${room.name}" class="room-card-img">
                    <div class="room-card-label">${room.name}</div>
                `;
                card.addEventListener('click', () => {
                    document.querySelectorAll('#rooms-list .room-card').forEach(c => c.classList.remove('active'));
                    card.classList.add('active');
                    loadRoom(room);
                });
                roomsContainer.appendChild(card);
            });
        }

        // Initialize dashboard and sidebar preset lists
        populateRoomsDashboard('all');
        populateRoomsSidebar('all');

        // Main Dashboard Room category filter tabs listeners
        document.querySelectorAll('.room-tab').forEach(tab => {
            tab.addEventListener('click', () => {
                document.querySelectorAll('.room-tab').forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                populateRoomsDashboard(tab.dataset.roomFilter);
            });
        });

        // Sidebar Presets tab category filters listeners
        document.querySelectorAll('[data-preset-filter]').forEach(tab => {
            tab.addEventListener('click', () => {
                document.querySelectorAll('[data-preset-filter]').forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                populateRoomsSidebar(tab.dataset.presetFilter);
            });
        });

        // 3. Populate Materials Grid in Sidebar
        populateMaterials('all');

        // Material Catalog Filtering chips
        document.querySelectorAll('.filter-chip').forEach(chip => {
            chip.addEventListener('click', () => {
                document.querySelectorAll('.filter-chip').forEach(c => c.classList.remove('active'));
                chip.classList.add('active');
                populateMaterials(chip.dataset.filter);
            });
        });
    }

    // --- Liked (favourite) tiles — persisted per browser ---
    function loadLikedTiles() {
        try { return new Set(JSON.parse(localStorage.getItem('vr_liked_tiles') || '[]')); }
        catch (e) { return new Set(); }
    }
    let likedTiles = loadLikedTiles();
    function saveLikedTiles() {
        try { localStorage.setItem('vr_liked_tiles', JSON.stringify([...likedTiles])); } catch (e) {}
    }
    function toggleLike(id) {
        if (likedTiles.has(id)) likedTiles.delete(id); else likedTiles.add(id);
        saveLikedTiles();
    }

    let currentMaterialFilter = 'all';

    function populateMaterials(filter) {
        currentMaterialFilter = filter || currentMaterialFilter;
        const list = document.getElementById('materials-list');
        if (!list) return;

        list.innerHTML = '';
        let filtered;
        if (currentMaterialFilter === 'all') filtered = materialsData;
        else if (currentMaterialFilter === 'liked') filtered = materialsData.filter(m => likedTiles.has(m.id));
        else filtered = materialsData.filter(m => m.type === currentMaterialFilter);

        if (!filtered.length) {
            list.innerHTML = `<div class="area-sub" style="grid-column:1/-1;padding:12px 0;">${
                currentMaterialFilter === 'liked' ? 'No liked tiles yet — tap the ♥ on any tile.' : 'No tiles in this category.'}</div>`;
            return;
        }

        filtered.forEach(mat => {
            const card = document.createElement('div');
            card.className = `product-card ${activeTile && mat.id === activeTile.id ? 'active' : ''}`;
            card.dataset.id = mat.id;
            card.innerHTML = `
                <div class="product-card-img-wrapper">
                    <img src="${mat.img}" alt="${mat.name}" class="product-card-img">
                    <button class="product-card-like ${likedTiles.has(mat.id) ? 'liked' : ''}" title="Like this tile" aria-label="Like">${likedTiles.has(mat.id) ? '♥' : '♡'}</button>
                </div>
                <div class="product-card-info">
                    <div>
                        <div class="product-card-brand">${mat.brand}</div>
                        <div class="product-card-name">${mat.name}</div>
                    </div>
                    <div class="product-card-footer">
                        <div class="product-card-price">₹${mat.price}/sq.ft</div>
                        <button class="product-card-apply">Apply</button>
                    </div>
                </div>
            `;

            card.querySelector('.product-card-like').addEventListener('click', (e) => {
                e.stopPropagation();
                toggleLike(mat.id);
                const b = e.currentTarget;
                const on = likedTiles.has(mat.id);
                b.classList.toggle('liked', on);
                b.textContent = on ? '♥' : '♡';
                if (currentMaterialFilter === 'liked' && !on) populateMaterials('liked');
            });

            card.addEventListener('click', () => {
                document.querySelectorAll('.product-card').forEach(c => c.classList.remove('active'));
                card.classList.add('active');
                selectTile(mat);
            });

            list.appendChild(card);
        });
    }

    // --- Visualizer Logic ---
    async function loadRoom(room) {
        activeRoom = room;
        document.getElementById('workspace-room-name').textContent = room.name;
        document.getElementById('workspace-room-sub').textContent = room.sub;
        
        // Synced presets state selection highlight inside Workspace Tab
        document.querySelectorAll('#rooms-list .room-card').forEach(c => {
            if (c.dataset.id === room.id) c.classList.add('active');
            else c.classList.remove('active');
        });

        // Show Loading Overlay
        const loading = document.getElementById('workspace-loading');
        if (loading) loading.style.display = 'flex';
        
        try {
            await new Promise((resolve, reject) => {
                imageCache.roomImage.onload = resolve;
                imageCache.roomImage.onerror = reject;
                imageCache.roomImage.src = room.img;
            });
            
            resizeCanvasToFit();
            renderVisualizer();
            
            // Upload current preset image to Python backend
            const response = await fetch(room.img);
            const blob = await response.blob();
            
            // Cache room blob for /api/visualize calls
            cachedRoomBlob = blob;
            cachedFloorMaskBlob = null;
            cachedWallMaskBlob = null;
            cachedWallFgBlob = null;
            visualizerState.wallAddBlob = null;
            visualizerState.wallRemoveBlob = null;
            if (typeof resetManualWallMasks === "function") resetManualWallMasks();
            visualizerState.serverRenderedImage = null;
            
            const formData = new FormData();
            formData.append('file', blob, 'room.png');
            
            const apiRes = await fetch('http://127.0.0.1:8000/api/segment', {
                method: 'POST',
                body: formData
            });
            
            if (!apiRes.ok) throw new Error('API segment call failed');
            
            const data = await apiRes.json();
            
            // Load transparent floor and wall masks
            const loadFloorMask = new Promise((resolve, reject) => {
                imageCache.maskImage.onload = resolve;
                imageCache.maskImage.onerror = reject;
                imageCache.maskImage.src = data.floor_mask;
            });
            
            const loadWallMask = new Promise((resolve, reject) => {
                imageCache.wallMaskImage.onload = resolve;
                imageCache.wallMaskImage.onerror = reject;
                imageCache.wallMaskImage.src = data.wall_mask;
            });
            
            await Promise.all([loadFloorMask, loadWallMask]);
            
            visualizerState.maskDataUrl = data.floor_mask;
            visualizerState.polygon = data.floor_polygon;
            visualizerState.floorQuad = data.floor_quad || [];
            visualizerState.wallMaskDataUrl = data.wall_mask;
            visualizerState.wallPolygon = data.wall_polygon;
            visualizerState.wallQuads = data.wall_quads || [];
            visualizerState.floorArea = data.floor_area || null;
            visualizerState.wallArea = data.wall_area || null;
            visualizerState.detectedObstacles = data.detected_obstacles || {};
            visualizerState.pixelsPerMeter = data.pixels_per_meter || 0;
            visualizerState.perspective = data.perspective || null;

            // Cache mask blobs for /api/visualize calls
            cachedFloorMaskBlob = dataUrlToBlob(data.floor_mask);
            cachedWallMaskBlob = dataUrlToBlob(data.wall_mask);
            cachedWallFgBlob = data.wall_fg_mask ? dataUrlToBlob(data.wall_fg_mask) : null;
            
            // Update area information panel
            updateAreaPanel();
            updateCoveragePanel();

            // Render outline for active target
            const activePolygon = visualizerState.target === 'floor' ? data.floor_polygon : data.wall_polygon;
            drawOutlineSVG(activePolygon, data.width, data.height);
            
            if (activeFloorTile || activeWallTile) {
                requestServerVisualization();
            } else {
                renderVisualizer();
            }
            
        } catch (err) {
            console.error('Error loading room or segmentation:', err);
            renderVisualizer();
        } finally {
            if (loading) loading.style.display = 'none';
        }
    }

    function selectTile(mat) {
        if (visualizerState.target === 'floor') {
            activeFloorTile = mat;
            visualizerState.activeFloorTileId = mat.id;
            activeTile = activeFloorTile;
            
            if (imageCache.tileImage.complete && imageCache.tileImage.src && imageCache.tileImage.src.endsWith(mat.img)) {
                requestServerVisualization();
            } else {
                imageCache.tileImage.onload = () => {
                    requestServerVisualization();
                };
                imageCache.tileImage.src = mat.img;
            }
        } else {
            activeWallTile = mat;
            visualizerState.activeWallTileId = mat.id;
            activeTile = activeWallTile;
            
            if (imageCache.wallTileImage.complete && imageCache.wallTileImage.src && imageCache.wallTileImage.src.endsWith(mat.img)) {
                requestServerVisualization();
            } else {
                imageCache.wallTileImage.onload = () => {
                    requestServerVisualization();
                };
                imageCache.wallTileImage.src = mat.img;
            }
        }

        
        // Update Bottom selected swatch footer
        const currentActive = visualizerState.target === 'floor' ? activeFloorTile : activeWallTile;
        if (currentActive) {
            document.getElementById('selected-material-img').src = currentActive.img;
            document.getElementById('selected-material-brand').textContent = currentActive.brand;
            document.getElementById('selected-material-name').textContent = currentActive.name;
            document.getElementById('selected-material-price').textContent = `₹${currentActive.price} / sq.ft`;
        } else {
            document.getElementById('selected-material-img').src = 'assets/placeholder.png';
            document.getElementById('selected-material-brand').textContent = '-';
            document.getElementById('selected-material-name').textContent = 'No wall tile selected';
            document.getElementById('selected-material-price').textContent = '';
        }
        
        // Highlight corresponding product card (synced)
        const activeId = visualizerState.target === 'floor' ? 
            (activeFloorTile ? activeFloorTile.id : null) : 
            (activeWallTile ? activeWallTile.id : null);
            
        document.querySelectorAll('.product-card').forEach(c => {
            if (c.dataset.id === activeId) c.classList.add('active');
            else c.classList.remove('active');
        });
    }
    
    // ============================================================
    // SERVER-SIDE VISUALIZATION (Option B) 
    // ============================================================
    
    // Show canvas loading shimmer while waiting for server
    function showCanvasShimmer(show) {
        const shimmer = document.getElementById('canvas-shimmer');
        if (shimmer) shimmer.style.display = show ? 'flex' : 'none';
    }
    
    // Core function: POST room + separate floor_tile & wall_tile + params to /api/visualize
    async function requestServerVisualization() {
        if (!cachedRoomBlob) {
            console.warn('[Visualize] No cached room blob yet.');
            renderVisualizer();
            return;
        }
        
        const floorActive = activeFloorTile && cachedFloorMaskBlob;
        const wallActive  = activeWallTile  && cachedWallMaskBlob;
        
        if (!floorActive && !wallActive) {
            visualizerState.serverRenderedImage = null;
            renderVisualizer();
            return;
        }
        
        showCanvasShimmer(true);
        
        try {
            const fd = new FormData();
            fd.append('room', cachedRoomBlob, 'room.jpg');
            
            // ── Attach Floor Tile if active ───────────────────────
            if (floorActive) {
                const floorTileBlob = await imageToBlob(imageCache.tileImage);
                if (floorTileBlob) {
                    fd.append('floor_tile', floorTileBlob, 'floor_tile.jpg');
                    fd.append('floor_scale',           visualizerState.scale);
                    fd.append('floor_rotation',        visualizerState.rotation);
                    fd.append('floor_brightness',      visualizerState.brightness);
                    fd.append('floor_pattern',         visualizerState.pattern);
                    fd.append('floor_grout_width',     visualizerState.groutWidth);
                    fd.append('floor_grout_color',     visualizerState.groutColor);
                    fd.append('floor_finish',          visualizerState.finish);
                    fd.append('floor_shadow_strength', visualizerState.shadowStrength);
                    fd.append('floor_tiles_x',         visualizerState.tilesX);
                    fd.append('floor_tiles_y',         visualizerState.tilesY);
                    const fmm = parseTileMM(activeFloorTile && activeFloorTile.specs);
                    fd.append('floor_tile_wmm',        fmm.w);
                    fd.append('floor_tile_hmm',        fmm.h);
                    fd.append('floor_slab',            visualizerState.slab ? 1 : 0);
                }
            }
            
            // ── Attach Wall Tile if active ────────────────────────
            if (wallActive) {
                const wallTileBlob = await imageToBlob(imageCache.wallTileImage);
                if (wallTileBlob) {
                    fd.append('wall_tile', wallTileBlob, 'wall_tile.jpg');
                    fd.append('wall_scale',           visualizerState.wallScale);
                    fd.append('wall_rotation',        visualizerState.wallRotation);
                    fd.append('wall_brightness',      visualizerState.wallBrightness);
                    fd.append('wall_pattern',         visualizerState.wallPattern);
                    fd.append('wall_grout_width',     visualizerState.wallGroutWidth);
                    fd.append('wall_grout_color',     visualizerState.wallGroutColor);
                    fd.append('wall_finish',          visualizerState.wallFinish);
                    fd.append('wall_shadow_strength', visualizerState.wallShadowStrength);
                    fd.append('wall_tiles_x',         visualizerState.wallTilesX);
                    fd.append('wall_tiles_y',         visualizerState.wallTilesY);
                    const wmm = parseTileMM(activeWallTile && activeWallTile.specs);
                    fd.append('wall_tile_wmm',         wmm.w);
                    fd.append('wall_tile_hmm',         wmm.h);
                    fd.append('wall_slab',             visualizerState.wallSlab ? 1 : 0);
                }
            }
            
            // ── Masks & Quads ─────────────────────────────────────
            if (cachedFloorMaskBlob) fd.append('floor_mask', cachedFloorMaskBlob, 'floor_mask.png');
            if (cachedWallMaskBlob)  fd.append('wall_mask',  cachedWallMaskBlob,  'wall_mask.png');
            if (cachedWallFgBlob)    fd.append('wall_fg_mask', cachedWallFgBlob, 'wall_fg.png');
            if (visualizerState.wallAddBlob)    fd.append('wall_add_mask', visualizerState.wallAddBlob, 'wall_add.png');
            if (visualizerState.wallRemoveBlob) fd.append('wall_remove_mask', visualizerState.wallRemoveBlob, 'wall_remove.png');
            fd.append('floor_quad',  JSON.stringify(visualizerState.floorQuad));
            fd.append('wall_quads',  JSON.stringify(visualizerState.wallQuads));
            fd.append('pixels_per_meter', visualizerState.pixelsPerMeter || 0);
            
            const res = await fetch('http://127.0.0.1:8000/api/visualize', {
                method: 'POST',
                body: fd,
            });
            
            if (!res.ok) throw new Error(`/api/visualize returned ${res.status}`);
            const data = await res.json();
            
            // Cache and display
            visualizerState.serverRenderedImage = data.image;
            renderVisualizer();
            
        } catch (err) {
            console.error('[Visualize] Server render failed, falling back to client render:', err);
            visualizerState.serverRenderedImage = null;
            renderVisualizer();
        } finally {
            showCanvasShimmer(false);
        }
    }
    
    // Debounced wrapper — prevents hammering server on every slider tick
    function debouncedVisualize(delayMs = 280) {
        clearTimeout(visualizeDebounceTimer);
        visualizeDebounceTimer = setTimeout(requestServerVisualization, delayMs);
        if (typeof updateCoveragePanel === 'function') updateCoveragePanel();
    }
    
    // Helper: canvas Image element → Blob
    function imageToBlob(imgEl) {
        return new Promise((resolve) => {
            try {
                const tmp = document.createElement('canvas');
                tmp.width  = imgEl.naturalWidth  || 256;
                tmp.height = imgEl.naturalHeight || 256;
                tmp.getContext('2d').drawImage(imgEl, 0, 0);
                tmp.toBlob(resolve, 'image/jpeg', 0.9);
            } catch (e) {
                resolve(null);
            }
        });
    }
    
    // Helper: data URL → Blob (synchronous & robust)
    function dataUrlToBlob(dataUrl) {
        if (!dataUrl) return null;
        try {
            const parts = dataUrl.split(',');
            const mimeMatch = parts[0].match(/:(.*?);/);
            const mime = mimeMatch ? mimeMatch[1] : 'image/png';
            const bstr = atob(parts[1]);
            let n = bstr.length;
            const u8arr = new Uint8Array(n);
            while (n--) {
                u8arr[n] = bstr.charCodeAt(n);
            }
            return new Blob([u8arr], { type: mime });
        } catch (e) {
            console.error('[Blob] dataUrlToBlob failed:', e);
            return null;
        }
    }

    
    // Area info panel updater
    function updateAreaPanel() {
        const panel = document.getElementById('area-info-panel');
        if (!panel) return;
        
        const fa = visualizerState.floorArea;
        const wa = visualizerState.wallArea;
        const obs = visualizerState.detectedObstacles;
        
        let html = '';
        
        if (fa) {
            const obstacleList = Object.entries(fa.obstacles || {});
            html += `
                <div class="area-row">
                    <span class="area-label">🟧 Floor Area</span>
                    <span class="area-value">${fa.net_sqft} sq.ft</span>
                </div>
                <div class="area-sub">Total: ${fa.total_sqft} sq.ft</div>`;
            if (obstacleList.length > 0) {
                html += `<div class="area-sub area-obstacles">Excluding: ${obstacleList.map(([k,v]) => `${k} (${v} sq.ft)`).join(', ')}</div>`;
            }
        }
        if (wa) {
            const obstacleList = Object.entries(wa.obstacles || {});
            html += `
                <div class="area-row" style="margin-top:8px">
                    <span class="area-label">🟦 Wall Area</span>
                    <span class="area-value">${wa.net_sqft} sq.ft</span>
                </div>
                <div class="area-sub">Total: ${wa.total_sqft} sq.ft</div>`;
            if (obstacleList.length > 0) {
                html += `<div class="area-sub area-obstacles">Excluding: ${obstacleList.map(([k,v]) => `${k} (${v} sq.ft)`).join(', ')}</div>`;
            }
        }
        
        const obsKeys = Object.keys(obs);
        if (obsKeys.length > 0) {
            html += `<div class="area-sub area-obstacles" style="margin-top:8px">🔍 Detected: ${obsKeys.join(', ')}</div>`;
        }
        
        if (!html) html = '<div class="area-sub">Area data will appear after room is analyzed.</div>';
        panel.innerHTML = html;
    }

    function resizeCanvasToFit() {
        const container = canvas.parentElement;
        if (!container) return;
        
        const rect = container.getBoundingClientRect();
        const width = rect.width;
        const height = rect.width * 0.75;
        
        canvas.width = width;
        canvas.height = height;
        
        offscreenCanvas.width = width;
        offscreenCanvas.height = height;
        
        compareCanvasBefore.width = width;
        compareCanvasBefore.height = height;
        compareCanvasAfter.width = width;
        compareCanvasAfter.height = height;
        
        const wrapper = document.querySelector('.split-slider-wrapper');
        if (wrapper) {
            wrapper.style.height = `${height}px`;
        }
    }

    // Generic manual loop-based canvas tiling renderer supporting Floor and Wall targets
    function drawTilingPattern(oCtx, tileImg, W, H, target) {
        oCtx.save();
        oCtx.clearRect(0, 0, W, H);
        
        const isFloor = target === 'floor';
        const currentScale = isFloor ? visualizerState.scale : visualizerState.wallScale;
        const currentRotation = isFloor ? visualizerState.rotation : visualizerState.wallRotation;
        const currentBrightness = isFloor ? visualizerState.brightness : visualizerState.wallBrightness;
        const currentPattern = isFloor ? visualizerState.pattern : visualizerState.wallPattern;
        const currentGroutWidth = isFloor ? visualizerState.groutWidth : visualizerState.wallGroutWidth;
        const currentGroutColor = isFloor ? visualizerState.groutColor : visualizerState.wallGroutColor;
        
        // Set brightness filter
        oCtx.filter = `brightness(${currentBrightness}%)`;
        
        // Set center translations and rotation angles
        oCtx.translate(W / 2, H / 2);
        oCtx.rotate(currentRotation * Math.PI / 180);
        
        const baseScale = activeRoom.baseScale || 1.0;
        const tileSizePx = W * currentScale * baseScale;
        const tileW = tileSizePx;
        const tileH = tileSizePx;
        
        const maxRange = Math.max(W, H) * 2;
        
        for (let y = -maxRange; y < maxRange; y += tileH) {
            let rowOffset = 0;
            if (currentPattern === 'brick' && Math.round(y / tileH) % 2 !== 0) {
                rowOffset = tileW / 2;
            }
            
            for (let x = -maxRange; x < maxRange; x += tileW) {
                const rx = x + rowOffset;
                oCtx.drawImage(tileImg, rx, y, tileW, tileH);
                if (currentGroutWidth > 0) {
                    oCtx.strokeStyle = currentGroutColor;
                    oCtx.lineWidth = currentGroutWidth;
                    oCtx.strokeRect(rx, y, tileW, tileH);
                }
            }
        }
        oCtx.restore();
    }

    // Draw a source triangle (s0..s2) onto ctx mapped to destination triangle (d0..d2).
    // Used to piece-wise warp the flat tile pattern into the detected floor quad
    // so the instant fallback preview also foreshortens with depth.
    function drawImageTriangle(dctx, img, s, d) {
        dctx.save();
        dctx.beginPath();
        dctx.moveTo(d[0], d[1]); dctx.lineTo(d[2], d[3]); dctx.lineTo(d[4], d[5]);
        dctx.closePath();
        dctx.clip();
        const x0 = s[0], y0 = s[1], x1 = s[2], y1 = s[3], x2 = s[4], y2 = s[5];
        const dx0 = d[0], dy0 = d[1], dx1 = d[2], dy1 = d[3], dx2 = d[4], dy2 = d[5];
        const den = x0 * (y1 - y2) - x1 * (y0 - y2) + x2 * (y0 - y1);
        if (Math.abs(den) < 1e-6) { dctx.restore(); return; }
        const a = (dx0 * (y1 - y2) - dx1 * (y0 - y2) + dx2 * (y0 - y1)) / den;
        const b = (dy0 * (y1 - y2) - dy1 * (y0 - y2) + dy2 * (y0 - y1)) / den;
        const c = (x0 * (dx1 - dx2) - x1 * (dx0 - dx2) + x2 * (dx0 - dx1)) / den;
        const dd = (x0 * (dy1 - dy2) - x1 * (dy0 - dy2) + x2 * (dy0 - dy1)) / den;
        const e = (x0 * (y1 * dx2 - y2 * dx1) - x1 * (y0 * dx2 - y2 * dx0) + x2 * (y0 * dx1 - y1 * dx0)) / den;
        const f = (x0 * (y1 * dy2 - y2 * dy1) - x1 * (y0 * dy2 - y2 * dy0) + x2 * (y0 * dy1 - y1 * dy0)) / den;
        dctx.transform(a, b, c, dd, e, f);
        dctx.drawImage(img, 0, 0);
        dctx.restore();
    }

    // Warp a fronto-parallel tile canvas into the floor quad (canvas-space),
    // using horizontal strips for a smooth perspective approximation.
    function warpToFloorQuad(srcCanvas, W, H) {
        const q = visualizerState.floorQuad;
        if (!q || q.length < 4) return srcCanvas;
        const nW = imageCache.roomImage.naturalWidth || W;
        const nH = imageCache.roomImage.naturalHeight || H;
        const kx = W / nW, ky = H / nH;
        const P = q.map(p => [p[0] * kx, p[1] * ky]);  // [farL, farR, nearR, nearL]
        const out = document.createElement('canvas');
        out.width = W; out.height = H;
        const o = out.getContext('2d');
        const N = 40, sW = srcCanvas.width, sH = srcCanvas.height;
        for (let i = 0; i < N; i++) {
            const t0 = i / N, t1 = (i + 1) / N;
            const lerp = (a, b, t) => a + (b - a) * t;
            const lx0 = lerp(P[3][0], P[0][0], t0), lx1 = lerp(P[3][0], P[0][0], t1);
            const rx0 = lerp(P[2][0], P[1][0], t0), rx1 = lerp(P[2][0], P[1][0], t1);
            const yy0 = lerp(P[3][1], P[0][1], t0), yy1 = lerp(P[3][1], P[0][1], t1);
            const sYa = (1 - t1) * sH, sYb = (1 - t0) * sH;
            drawImageTriangle(o, srcCanvas,
                [0, sYa, sW, sYa, 0, sYb], [lx1, yy1, rx1, yy1, lx0, yy0]);
            drawImageTriangle(o, srcCanvas,
                [sW, sYa, sW, sYb, 0, sYb], [rx1, yy1, rx0, yy0, lx0, yy0]);
        }
        return out;
    }

    // Core workspace compositor — draws server-rendered image when available,
    // falls back to client-side canvas render for speed while waiting.
    function renderVisualizer() {
        const W = canvas.width;
        const H = canvas.height;
        if (!W || !H || !imageCache.roomImage.complete) return;
        
        ctx.clearRect(0, 0, W, H);
        ctx.drawImage(imageCache.roomImage, 0, 0, W, H);
        
        // If we have a server-rendered composite image, draw that on top
        if (visualizerState.serverRenderedImage) {
            const serverImg = new Image();
            serverImg.onload = () => {
                ctx.clearRect(0, 0, W, H);
                ctx.drawImage(serverImg, 0, 0, W, H);
                if (visualizerState.mode === 'compare') {
                    updateCompareCanvases();
                }
            };
            serverImg.src = visualizerState.serverRenderedImage;
            // Hide selection outline — tile is applied, no need for the dashed polygon
            const outlineSvg = document.getElementById('outline-svg');
            if (outlineSvg) outlineSvg.style.display = 'none';
            return; // early return — the async onload will finish the paint
        }
        
        // No server render yet — restore the outline so the user can see the target area
        const outlineSvg = document.getElementById('outline-svg');
        if (outlineSvg) outlineSvg.style.display = '';

        
        // ── Fallback: client-side flat tiling (shown instantly before server responds) ──
        const shadowCanvas = document.createElement('canvas');
        shadowCanvas.width = W;
        shadowCanvas.height = H;
        const shadowCtx = shadowCanvas.getContext('2d');
        
        const blurRadius = Math.max(5, Math.round(W * 0.035));
        shadowCtx.filter = `grayscale(100%) blur(${blurRadius}px) brightness(1.15)`;
        shadowCtx.drawImage(imageCache.roomImage, -blurRadius, -blurRadius, W + 2 * blurRadius, H + 2 * blurRadius);
        
        // 1. Draw floor tiling (warped into the detected floor quad for perspective)
        if (activeFloorTile && visualizerState.maskDataUrl && imageCache.maskImage.complete && imageCache.tileImage.complete) {
            drawTilingPattern(offCtx, imageCache.tileImage, W, H, 'floor');

            const warped = warpToFloorQuad(offscreenCanvas, W, H);
            offCtx.save();
            offCtx.setTransform(1, 0, 0, 1, 0, 0);
            offCtx.globalCompositeOperation = 'source-over';
            offCtx.clearRect(0, 0, W, H);
            offCtx.drawImage(warped, 0, 0, W, H);

            offCtx.globalCompositeOperation = 'multiply';
            offCtx.drawImage(shadowCanvas, 0, 0, W, H);

            offCtx.globalCompositeOperation = 'destination-in';
            offCtx.drawImage(imageCache.maskImage, 0, 0, W, H);
            offCtx.restore();

            ctx.drawImage(offscreenCanvas, 0, 0, W, H);
        }
        
        // 2. Draw wall tiling
        if (activeWallTile && visualizerState.wallMaskDataUrl && imageCache.wallMaskImage.complete && imageCache.wallTileImage.complete) {
            drawTilingPattern(offCtx, imageCache.wallTileImage, W, H, 'wall');
            
            offCtx.save();
            offCtx.globalCompositeOperation = 'multiply';
            offCtx.drawImage(shadowCanvas, 0, 0, W, H);
            
            offCtx.globalCompositeOperation = 'destination-in';
            offCtx.drawImage(imageCache.wallMaskImage, 0, 0, W, H);
            offCtx.restore();
            
            ctx.drawImage(offscreenCanvas, 0, 0, W, H);
        }
        
        // 3. Glossy floor reflection overlay
        if (visualizerState.finish === 'glossy' && visualizerState.maskDataUrl && imageCache.maskImage.complete) {
            ctx.save();
            ctx.globalAlpha = 0.12;
            ctx.globalCompositeOperation = 'screen';
            
            const glossCanvas = document.createElement('canvas');
            glossCanvas.width = W;
            glossCanvas.height = H;
            const glossCtx = glossCanvas.getContext('2d');
            
            glossCtx.drawImage(imageCache.roomImage, 0, 0, W, H);
            glossCtx.globalCompositeOperation = 'destination-in';
            glossCtx.drawImage(imageCache.maskImage, 0, 0, W, H);
            
            ctx.drawImage(glossCanvas, 0, 0, W, H);
            ctx.restore();
        }
        
        // Sync compare canvases
        if (visualizerState.mode === 'compare') {
            updateCompareCanvases();
        }
    }

    // Draw outline SVGs on overlay
    function drawOutlineSVG(points, originalW, originalH) {
        if (!outlineSvg) return;
        
        outlineSvg.innerHTML = '';
        if (!points || points.length === 0) return;
        
        outlineSvg.setAttribute('viewBox', `0 0 ${originalW} ${originalH}`);
        
        const polygon = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
        const pointsString = points.map(p => p.join(',')).join(' ');
        
        polygon.setAttribute('points', pointsString);
        polygon.setAttribute('class', 'outline-poly-path');
        
        polygon.addEventListener('mouseenter', () => {
            const hotspot = document.getElementById('hotspot-floor');
            if (hotspot) hotspot.classList.add('hotspot-hover');
        });
        polygon.addEventListener('mouseleave', () => {
            const hotspot = document.getElementById('hotspot-floor');
            if (hotspot) hotspot.classList.remove('hotspot-hover');
        });
        
        outlineSvg.appendChild(polygon);
    }

    // --- Draggable Compare View Modes ---
    function toggleCompareMode(enable) {
        const designContainer = document.getElementById('canvas-container-design');
        const compareContainer = document.getElementById('canvas-container-compare');
        
        if (enable) {
            visualizerState.mode = 'compare';
            designContainer.style.display = 'none';
            compareContainer.style.display = 'block';
            
            document.getElementById('mode-design-btn').classList.remove('active');
            document.getElementById('mode-compare-btn').classList.add('active');
            
            updateCompareCanvases();
            initCompareSlider();
        } else {
            visualizerState.mode = 'design';
            designContainer.style.display = 'block';
            compareContainer.style.display = 'none';
            
            document.getElementById('mode-design-btn').classList.add('active');
            document.getElementById('mode-compare-btn').classList.remove('active');
        }
    }

    function updateCompareCanvases() {
        const W = canvas.width;
        const H = canvas.height;
        if (!W || !H) return;

        // Layer geometry: the "after-wrapper" clips the LEFT part and holds
        // compareCanvasAfter; compareCanvasBefore shows through on the RIGHT.
        const originalLeft = visualizerState.compareOriginalSide === 'left';
        const leftImg  = originalLeft ? imageCache.roomImage : canvas;   // shown left
        const rightImg = originalLeft ? canvas : imageCache.roomImage;   // shown right

        compareCtxAfter.clearRect(0, 0, W, H);
        compareCtxAfter.drawImage(leftImg, 0, 0, W, H);
        compareCtxBefore.clearRect(0, 0, W, H);
        compareCtxBefore.drawImage(rightImg, 0, 0, W, H);
    }

    function initCompareSlider() {
        compareAfterWrapper.style.width = '50%';
        splitSliderBar.style.left = '50%';
        
        compareCanvasAfter.style.width = `${canvas.width}px`;
        compareCanvasAfter.style.height = `${canvas.height}px`;

        let isDragging = false;
        
        function moveSlider(clientX) {
            const rect = compareCanvasBefore.getBoundingClientRect();
            const x = clientX - rect.left;
            let percentage = (x / rect.width) * 100;
            percentage = Math.max(0, Math.min(percentage, 100));
            
            compareAfterWrapper.style.width = `${percentage}%`;
            splitSliderBar.style.left = `${percentage}%`;
        }

        const sliderWrapper = document.querySelector('.split-slider-wrapper');
        sliderWrapper.addEventListener('mousedown', (e) => {
            isDragging = true;
            moveSlider(e.clientX);
        });

        window.addEventListener('mousemove', (e) => {
            if (!isDragging) return;
            moveSlider(e.clientX);
        });

        window.addEventListener('mouseup', () => {
            isDragging = false;
        });

        sliderWrapper.addEventListener('touchstart', (e) => {
            isDragging = true;
            if (e.touches && e.touches[0]) {
                moveSlider(e.touches[0].clientX);
            }
        });

        window.addEventListener('touchmove', (e) => {
            if (!isDragging) return;
            if (e.touches && e.touches[0]) {
                moveSlider(e.touches[0].clientX);
            }
        });

        window.addEventListener('touchend', () => {
            isDragging = false;
        });
    }

    // --- Exporter Functionalities ---
    
    // 1. Download JPEG Screenshot
    const screenshotBtn = document.getElementById('toolbar-screenshot');
    if (screenshotBtn) {
        screenshotBtn.addEventListener('click', () => {
            const dataUrl = canvas.toDataURL('image/jpeg', 0.95);
            const link = document.createElement('a');
            link.download = `room-design-${activeRoom.id}-${activeTile.id}.jpg`;
            link.href = dataUrl;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        });
    }

    // 2. Export PDF Catalog / Design Sheet
    const exportCatalogBtn = document.getElementById('toolbar-export-catalog');
    if (exportCatalogBtn) {
        exportCatalogBtn.addEventListener('click', exportDesignCatalogSheet);
    }

    function exportDesignCatalogSheet() {
        // Create a large A4 vertical canvas (1240 x 1754 px for nice resolution)
        const pdfCanvas = document.createElement('canvas');
        pdfCanvas.width = 1200;
        pdfCanvas.height = 1600;
        const pCtx = pdfCanvas.getContext('2d');

        // Background
        pCtx.fillStyle = '#ffffff';
        pCtx.fillRect(0, 0, 1200, 1600);

        // Header
        pCtx.fillStyle = '#0f172a'; // slate 900
        pCtx.font = 'bold 36px "Plus Jakarta Sans"';
        pCtx.fillText('VISIONROOM AI', 80, 100);
        
        pCtx.fillStyle = '#64748b'; // slate 500
        pCtx.font = '600 16px "Plus Jakarta Sans"';
        pCtx.fillText('AUTOMATED MATERIALS SPECIFICATION & DESIGN SHEET', 80, 130);

        // Header Line divider
        pCtx.strokeStyle = '#e2e8f0';
        pCtx.lineWidth = 2;
        pCtx.beginPath();
        pCtx.moveTo(80, 160);
        pCtx.lineTo(1120, 160);
        pCtx.stroke();

        // 1. Draw Final Rendered Room Image (scale it to fit ~1040 width)
        const renderW = 1040;
        const renderH = 780; // 4:3
        pCtx.drawImage(canvas, 80, 200, renderW, renderH);

        // Border around room render
        pCtx.strokeStyle = '#cbd5e1';
        pCtx.strokeRect(80, 200, renderW, renderH);

        // 2. Applied Material Swatch section
        pCtx.fillStyle = '#f8fafc';
        pCtx.fillRect(80, 1030, 480, 450);
        pCtx.strokeRect(80, 1030, 480, 450);

        // Draw Swatch tile
        pCtx.drawImage(imageCache.tileImage, 120, 1070, 150, 150);
        pCtx.strokeStyle = '#cbd5e1';
        pCtx.strokeRect(120, 1070, 150, 150);

        // Material description labels
        pCtx.fillStyle = '#0f172a';
        pCtx.font = 'bold 22px "Plus Jakarta Sans"';
        pCtx.fillText(activeTile.brand, 120, 1260);

        pCtx.fillStyle = '#334155';
        pCtx.font = '600 18px "Plus Jakarta Sans"';
        pCtx.fillText(activeTile.name, 120, 1295);

        pCtx.fillStyle = '#10b981'; // emerald
        pCtx.font = 'bold 20px "Plus Jakarta Sans"';
        pCtx.fillText(`Material Cost: ₹${activeTile.price} / sq.ft`, 120, 1335);

        pCtx.fillStyle = '#64748b';
        pCtx.font = '13px "Plus Jakarta Sans"';
        pCtx.fillText('Applied Surface: Flooring', 120, 1380);
        pCtx.fillText(`Layout Finish: ${visualizerState.finish.toUpperCase()}`, 120, 1405);

        // 3. Technical Parameters / Specifications section
        pCtx.fillStyle = '#f8fafc';
        pCtx.fillRect(600, 1030, 520, 450);
        pCtx.strokeStyle = '#cbd5e1';
        pCtx.strokeRect(600, 1030, 520, 450);

        pCtx.fillStyle = '#0f172a';
        pCtx.font = 'bold 22px "Plus Jakarta Sans"';
        pCtx.fillText('Tiling Engine Alignment Parameters', 640, 1080);

        const specsTable = [
            ['Space Preset', activeRoom.name],
            ['Tiling Pattern', visualizerState.pattern === 'brick' ? 'Brick Offset (50%)' : 'Standard Grid'],
            ['Tiling Size (Scale)', `${visualizerState.scale.toFixed(2)}x`],
            ['Tiling Angle (Rotation)', `${visualizerState.rotation}°`],
            ['Grout Width', `${visualizerState.groutWidth}px`],
            ['Grout Line Color', visualizerState.groutColor],
            ['Generated Area Estimate', `${document.getElementById('result-area').textContent}`],
            ['Total Project Cost Est.', `${document.getElementById('result-total').textContent}`]
        ];

        let startY = 1140;
        specsTable.forEach(([key, val]) => {
            pCtx.fillStyle = '#64748b';
            pCtx.font = '600 15px "Plus Jakarta Sans"';
            pCtx.fillText(key, 640, startY);

            pCtx.fillStyle = '#0f172a';
            pCtx.font = 'bold 15px "Plus Jakarta Sans"';
            pCtx.fillText(val, 880, startY);

            // Row dividers
            pCtx.strokeStyle = '#e2e8f0';
            pCtx.lineWidth = 1;
            pCtx.beginPath();
            pCtx.moveTo(640, startY + 12);
            pCtx.lineTo(1080, startY + 12);
            pCtx.stroke();

            startY += 40;
        });

        // Footer note
        pCtx.fillStyle = '#94a3b8';
        pCtx.font = '600 13px "Plus Jakarta Sans"';
        pCtx.fillText('THIS SPECIFICATION SHEET IS AUTO-GENERATED BY VISIONROOM AI DESIGN TOOL FOR ROUGH LOGISTICS PLANNING.', 80, 1530);

        // Convert to dataUrl and trigger download
        const pdfDataUrl = pdfCanvas.toDataURL('image/png');
        const link = document.createElement('a');
        link.download = `VisionRoom-Catalog-${activeRoom.id}-${activeTile.id}.png`;
        link.href = pdfDataUrl;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }

    // --- Control Sliders Listeners ---
    
    // Helper to synchronize sliders UI when switching between Floor and Wall modes
    function syncSlidersUI() {
        const isFloor = visualizerState.target === 'floor';
        const currentScale = isFloor ? visualizerState.scale : visualizerState.wallScale;
        const currentRotation = isFloor ? visualizerState.rotation : visualizerState.wallRotation;
        const currentBrightness = isFloor ? visualizerState.brightness : visualizerState.wallBrightness;
        const currentGroutWidth = isFloor ? visualizerState.groutWidth : visualizerState.wallGroutWidth;
        const currentGroutColor = isFloor ? visualizerState.groutColor : visualizerState.wallGroutColor;
        const currentPattern = isFloor ? visualizerState.pattern : visualizerState.wallPattern;
        const currentFinish = isFloor ? visualizerState.finish : visualizerState.wallFinish;
        const currentSlab = isFloor ? visualizerState.slab : visualizerState.wallSlab;
        
        if (scaleSlider) {
            scaleSlider.value = currentScale;
            let label = 'Default';
            if (currentScale < 0.1) label = 'XS';
            else if (currentScale < 0.15) label = 'Small';
            else if (currentScale < 0.22) label = 'Default';
            else if (currentScale < 0.32) label = 'Large';
            else label = 'XL';
            scaleVal.textContent = label;
        }
        if (rotSlider) {
            rotSlider.value = currentRotation;
            rotVal.textContent = `${currentRotation}°`;
        }
        if (brightSlider) {
            brightSlider.value = currentBrightness;
            brightVal.textContent = `${currentBrightness}%`;
        }
        if (groutWidthSlider) {
            groutWidthSlider.value = currentGroutWidth;
            groutWidthVal.textContent = currentGroutWidth === 0 ? 'None' : `${currentGroutWidth}`;
        }

        document.querySelectorAll('.grout-swatch').forEach(s => {
            if (s.dataset.color === currentGroutColor) s.classList.add('active');
            else s.classList.remove('active');
        });
        document.querySelectorAll('.pattern-btn').forEach(b => {
            if (b.dataset.pattern === currentPattern) b.classList.add('active');
            else b.classList.remove('active');
        });
        document.querySelectorAll('.finish-btn').forEach(b => {
            if (b.dataset.finish === currentFinish) b.classList.add('active');
            else b.classList.remove('active');
        });
        document.querySelectorAll('.slab-btn').forEach(b => {
            if (parseInt(b.dataset.slab, 10) === (currentSlab || 0)) b.classList.add('active');
            else b.classList.remove('active');
        });
        
        // Swatch Footer
        const currentActive = isFloor ? activeFloorTile : activeWallTile;
        if (currentActive) {
            document.getElementById('selected-material-img').src = currentActive.img;
            document.getElementById('selected-material-brand').textContent = currentActive.brand;
            document.getElementById('selected-material-name').textContent = currentActive.name;
            document.getElementById('selected-material-price').textContent = `₹${currentActive.price} / sq.ft`;
        } else {
            document.getElementById('selected-material-img').src = 'assets/placeholder.png';
            document.getElementById('selected-material-brand').textContent = '-';
            document.getElementById('selected-material-name').textContent = 'No wall tile selected';
            document.getElementById('selected-material-price').textContent = '';
        }
        
        const activeId = isFloor ? (activeFloorTile ? activeFloorTile.id : null) : (activeWallTile ? activeWallTile.id : null);
        document.querySelectorAll('.product-card').forEach(c => {
            if (c.dataset.id === activeId) c.classList.add('active');
            else c.classList.remove('active');
        });
        
        const activePolygon = isFloor ? visualizerState.polygon : visualizerState.wallPolygon;
        const naturalW = imageCache.roomImage.naturalWidth || canvas.width;
        const naturalH = imageCache.roomImage.naturalHeight || canvas.height;
        drawOutlineSVG(activePolygon, naturalW, naturalH);
    }
    
    // Target Visualizer Button Listeners (Floor Tiling vs Wall Tiling)
    const targetFloorBtn = document.getElementById('target-floor-btn');
    const targetWallBtn = document.getElementById('target-wall-btn');
    if (targetFloorBtn && targetWallBtn) {
        targetFloorBtn.addEventListener('click', () => {
            targetFloorBtn.classList.add('active');
            targetWallBtn.classList.remove('active');
            visualizerState.target = 'floor';
            
            // Set header labels
            document.getElementById('catalog-header-title').textContent = 'Select Flooring';
            document.getElementById('catalog-header-desc').textContent = 'Choose a premium tile or planks from catalog';
            
            syncSlidersUI();
            renderVisualizer();
        });
        
        targetWallBtn.addEventListener('click', () => {
            targetWallBtn.classList.add('active');
            targetFloorBtn.classList.remove('active');
            visualizerState.target = 'wall';
            
            // Set header labels
            document.getElementById('catalog-header-title').textContent = 'Select Wall Coverings';
            document.getElementById('catalog-header-desc').textContent = 'Choose a premium tile or wallpaper for walls';
            
            syncSlidersUI();
            renderVisualizer();
        });
    }

    // Scale Size slider (controls tiles_x/tiles_y count)
    const scaleSlider = document.getElementById('adjust-scale');
    const scaleVal = document.getElementById('value-scale');
    if (scaleSlider && scaleVal) {
        scaleSlider.addEventListener('input', (e) => {
            const val = parseFloat(e.target.value);
            if (visualizerState.target === 'floor') {
                visualizerState.scale = val;
                // Map scale (0.06-0.4) to tilesX/Y
                visualizerState.tilesX = Math.max(2, Math.round(12 - val * 25));
                visualizerState.tilesY = Math.max(2, Math.round(10 - val * 20));
            } else {
                visualizerState.wallScale = val;
                visualizerState.wallTilesX = Math.max(2, Math.round(12 - val * 25));
                visualizerState.wallTilesY = Math.max(2, Math.round(10 - val * 20));
            }
            let label = 'Default';
            if (val < 0.1) label = 'XS';
            else if (val < 0.15) label = 'Small';
            else if (val < 0.22) label = 'Default';
            else if (val < 0.32) label = 'Large';
            else label = 'XL';
            scaleVal.textContent = label;
            debouncedVisualize();
        });
    }

    // Rotation Angle slider
    const rotSlider = document.getElementById('adjust-rotation');
    const rotVal = document.getElementById('value-rotation');
    if (rotSlider && rotVal) {
        rotSlider.addEventListener('input', (e) => {
            const val = parseInt(e.target.value, 10);
            if (visualizerState.target === 'floor') {
                visualizerState.rotation = val;
            } else {
                visualizerState.wallRotation = val;
            }
            rotVal.textContent = `${val}°`;
            debouncedVisualize();
        });
    }

    // Exposure Brightness slider
    const brightSlider = document.getElementById('adjust-brightness');
    const brightVal = document.getElementById('value-brightness');
    if (brightSlider && brightVal) {
        brightSlider.addEventListener('input', (e) => {
            const val = parseInt(e.target.value, 10);
            const bFloat = val / 100.0;
            if (visualizerState.target === 'floor') {
                visualizerState.brightness = bFloat;
            } else {
                visualizerState.wallBrightness = bFloat;
            }
            brightVal.textContent = `${val}%`;
            debouncedVisualize();
        });
    }

    // Shadow Strength slider (ambient light blend)
    const shadowSlider = document.getElementById('adjust-shadow');
    const shadowVal = document.getElementById('value-shadow');
    if (shadowSlider && shadowVal) {
        shadowSlider.addEventListener('input', (e) => {
            const val = parseInt(e.target.value, 10);
            const sFloat = val / 100.0;
            if (visualizerState.target === 'floor') {
                visualizerState.shadowStrength = sFloat;
            } else {
                visualizerState.wallShadowStrength = sFloat;
            }
            shadowVal.textContent = `${val}%`;
            debouncedVisualize();
        });
    }

    // Grout width slider
    const groutWidthSlider = document.getElementById('adjust-grout-width');
    const groutWidthVal = document.getElementById('value-grout-width');
    if (groutWidthSlider && groutWidthVal) {
        groutWidthSlider.addEventListener('input', (e) => {
            const val = parseInt(e.target.value, 10);
            if (visualizerState.target === 'floor') {
                visualizerState.groutWidth = val;
            } else {
                visualizerState.wallGroutWidth = val;
            }
            groutWidthVal.textContent = val === 0 ? 'None' : `${val}`;
            debouncedVisualize();
        });
    }

    // Grout Color swatch selectors
    document.querySelectorAll('.grout-swatch').forEach(swatch => {
        swatch.addEventListener('click', () => {
            document.querySelectorAll('.grout-swatch').forEach(s => s.classList.remove('active'));
            swatch.classList.add('active');
            if (visualizerState.target === 'floor') {
                visualizerState.groutColor = swatch.dataset.color;
            } else {
                visualizerState.wallGroutColor = swatch.dataset.color;
            }
            debouncedVisualize(150);
        });
    });

    // Pattern buttons Grid vs. Brick
    document.querySelectorAll('.pattern-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.pattern-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            if (visualizerState.target === 'floor') {
                visualizerState.pattern = btn.dataset.pattern;
            } else {
                visualizerState.wallPattern = btn.dataset.pattern;
            }
            debouncedVisualize(150);
        });
    });

    // Surface finish buttons Matte / Satin / Polished
    document.querySelectorAll('.finish-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.finish-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            if (visualizerState.target === 'floor') {
                visualizerState.finish = btn.dataset.finish;
            } else {
                visualizerState.wallFinish = btn.dataset.finish;
            }
            debouncedVisualize(150);
        });
    });

    // Surface style: Tiled vs Slab / Bookmatch
    document.querySelectorAll('.slab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.slab-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            const v = parseInt(btn.dataset.slab, 10);
            if (visualizerState.target === 'floor') {
                visualizerState.slab = v;
            } else {
                visualizerState.wallSlab = v;
            }
            debouncedVisualize(150);
        });
    });

    // "↺ Default" — restore the current surface's layout settings to factory
    const LAYOUT_DEFAULTS = {
        scale: 0.18, rotation: 0, brightness: 1.0, shadowStrength: 0.55,
        pattern: 'grid', groutWidth: 0, groutColor: '#ffffff', finish: 'matte', slab: 0,
    };
    function resetCurrentLayout() {
        const isFloor = visualizerState.target === 'floor';
        const d = LAYOUT_DEFAULTS;
        if (isFloor) {
            visualizerState.scale = d.scale; visualizerState.rotation = d.rotation;
            visualizerState.brightness = d.brightness; visualizerState.shadowStrength = d.shadowStrength;
            visualizerState.pattern = d.pattern; visualizerState.groutWidth = d.groutWidth;
            visualizerState.groutColor = d.groutColor; visualizerState.finish = d.finish;
            visualizerState.slab = d.slab; visualizerState.tilesX = 6; visualizerState.tilesY = 5;
        } else {
            visualizerState.wallScale = d.scale; visualizerState.wallRotation = d.rotation;
            visualizerState.wallBrightness = d.brightness; visualizerState.wallShadowStrength = d.shadowStrength;
            visualizerState.wallPattern = d.pattern; visualizerState.wallGroutWidth = d.groutWidth;
            visualizerState.wallGroutColor = d.groutColor; visualizerState.wallFinish = d.finish;
            visualizerState.wallSlab = d.slab; visualizerState.wallTilesX = 5; visualizerState.wallTilesY = 7;
        }
        if (brightSlider) brightSlider.value = 100;
        if (shadowSlider) shadowSlider.value = 55;
        syncSlidersUI();
        updateCoveragePanel();
        debouncedVisualize(120);
    }
    const resetLayoutBtn = document.getElementById('reset-layout-btn');
    if (resetLayoutBtn) resetLayoutBtn.addEventListener('click', resetCurrentLayout);

    // Reset controls
    const resetBtn = document.getElementById('toolbar-reset');
    if (resetBtn) {
        resetBtn.addEventListener('click', () => {
            // 1. Reset Floor parameters
            visualizerState.scale = 0.18;
            visualizerState.rotation = 0;
            visualizerState.brightness = 1.0;
            visualizerState.groutWidth = 0;
            visualizerState.groutColor = '#cccccc';
            visualizerState.pattern = 'grid';
            visualizerState.finish = 'matte';
            visualizerState.slab = 0;
            visualizerState.shadowStrength = 0.55;
            visualizerState.tilesX = 6;
            visualizerState.tilesY = 5;
            
            // 2. Reset Wall parameters
            visualizerState.wallScale = 0.18;
            visualizerState.wallRotation = 0;
            visualizerState.wallBrightness = 1.0;
            visualizerState.wallGroutWidth = 0;
            visualizerState.wallGroutColor = '#cccccc';
            visualizerState.wallPattern = 'grid';
            visualizerState.wallFinish = 'matte';
            visualizerState.wallSlab = 0;
            visualizerState.wallShadowStrength = 0.55;
            visualizerState.wallTilesX = 5;
            visualizerState.wallTilesY = 7;
            
            // 3. Clear selected tiles
            activeFloorTile = null;
            visualizerState.activeFloorTileId = null;
            imageCache.tileImage = new Image();
            
            activeWallTile = null;
            visualizerState.activeWallTileId = null;
            imageCache.wallTileImage = new Image();
            
            activeTile = null;
            
            // 4. Clear server-rendered image so original room photo is shown
            visualizerState.serverRenderedImage = null;
            clearTimeout(visualizeDebounceTimer);
            
            // 5. Update UI controls, swatches, and labels
            syncSlidersUI();
            
            // 6. Re-render visualizer canvas (original image only)
            renderVisualizer();
        });
    }

    // Toggle Workspace views
    const designModeBtn = document.getElementById('mode-design-btn');
    const compareModeBtn = document.getElementById('mode-compare-btn');
    if (designModeBtn) {
        designModeBtn.addEventListener('click', () => toggleCompareMode(false));
    }
    if (compareModeBtn) {
        compareModeBtn.addEventListener('click', () => toggleCompareMode(true));
    }

    // ── Wall detection debug view ────────────────────────────
    const wallDebugBtn = document.getElementById('mode-walldebug-btn');
    const wallDebugModal = document.getElementById('wall-debug-modal');
    const wallDebugClose = document.getElementById('wall-debug-close');
    async function openWallDebug() {
        if (!cachedRoomBlob) { alert('Load or upload a room first.'); return; }
        wallDebugModal.style.display = 'flex';
        document.getElementById('wall-debug-loading').style.display = 'block';
        document.getElementById('wall-debug-img').style.display = 'none';
        document.getElementById('wall-debug-metrics').textContent = '';
        try {
            const fd = new FormData();
            fd.append('room', cachedRoomBlob, 'room.jpg');
            if (activeWallTile && imageCache.wallTileImage) {
                const wb = await imageToBlob(imageCache.wallTileImage);
                if (wb) fd.append('wall_tile', wb, 'wall_tile.jpg');
                const wmm = parseTileMM(activeWallTile && activeWallTile.specs);
                fd.append('wall_tile_wmm', wmm.w);
                fd.append('wall_tile_hmm', wmm.h);
            }
            fd.append('wall_pattern', visualizerState.wallPattern || 'grid');
            fd.append('wall_finish', visualizerState.wallFinish || 'matte');
            const res = await fetch('http://127.0.0.1:8000/api/wall-debug', { method: 'POST', body: fd });
            const data = await res.json();
            if (data.status !== 'success') throw new Error(data.message || 'debug failed');
            const img = document.getElementById('wall-debug-img');
            img.src = data.panel;
            img.style.display = 'block';
            document.getElementById('wall-debug-loading').style.display = 'none';
            const m = data.metrics || {};
            document.getElementById('wall-debug-metrics').textContent =
                `surface coverage ${m.coverage}  ·  visible material ${m.visible_material}  ·  ` +
                `wall_conf ${m.wall_confidence}  ·  obj_conf ${m.object_confidence}  ·  ` +
                `depth ${m.depth_used ? 'on' : 'off'}  ·  planes ${(m.wall_planes || []).map(p => p.plane + '(' + p.confidence + ')').join(' ')}`;
        } catch (e) {
            document.getElementById('wall-debug-loading').textContent = 'Wall debug failed: ' + e.message;
        }
    }
    if (wallDebugBtn) wallDebugBtn.addEventListener('click', openWallDebug);
    if (wallDebugClose) wallDebugClose.addEventListener('click', () => { wallDebugModal.style.display = 'none'; });
    if (wallDebugModal) wallDebugModal.addEventListener('click', (e) => {
        if (e.target === wallDebugModal) wallDebugModal.style.display = 'none';
    });

    // ── Manual wall-mask brush (Phase H) ─────────────────────
    const maskEditBtn   = document.getElementById('mode-maskedit-btn');
    const maskEditBar    = document.getElementById('mask-edit-bar');
    const brushCanvas    = document.getElementById('mask-brush-canvas');
    const designWrapper  = document.getElementById('canvas-container-design');
    const brushSizeInput = document.getElementById('mask-brush-size');
    const maskEdit = { on: false, mode: 'add', size: 42, drawing: false, strokes: [] };
    let addMaskCanvas = null, removeMaskCanvas = null;

    function natSize() {
        return {
            w: imageCache.roomImage.naturalWidth || canvas.width,
            h: imageCache.roomImage.naturalHeight || canvas.height,
        };
    }
    function ensureMaskCanvases() {
        const { w, h } = natSize();
        if (!addMaskCanvas || addMaskCanvas.width !== w || addMaskCanvas.height !== h) {
            addMaskCanvas = document.createElement('canvas'); addMaskCanvas.width = w; addMaskCanvas.height = h;
            removeMaskCanvas = document.createElement('canvas'); removeMaskCanvas.width = w; removeMaskCanvas.height = h;
            maskEdit.strokes = [];
        }
    }
    function repaintBrushPreview() {
        const bctx = brushCanvas.getContext('2d');
        brushCanvas.width = canvas.width; brushCanvas.height = canvas.height;
        bctx.clearRect(0, 0, brushCanvas.width, brushCanvas.height);
        const { w, h } = natSize();
        const sx = brushCanvas.width / w, sy = brushCanvas.height / h;
        for (const s of maskEdit.strokes) {
            bctx.strokeStyle = s.mode === 'add' ? 'rgba(99,102,241,0.55)' : 'rgba(239,68,68,0.55)';
            bctx.lineWidth = s.size * sx;
            bctx.lineCap = 'round'; bctx.lineJoin = 'round';
            bctx.beginPath();
            s.pts.forEach((p, i) => { const x = p[0] * sx, y = p[1] * sy; i ? bctx.lineTo(x, y) : bctx.moveTo(x, y); });
            if (s.pts.length === 1) { bctx.lineTo(s.pts[0][0] * sx + 0.1, s.pts[0][1] * sy); }
            bctx.stroke();
        }
    }
    function commitStroke(stroke) {
        const target = stroke.mode === 'add' ? addMaskCanvas : removeMaskCanvas;
        const tctx = target.getContext('2d');
        tctx.strokeStyle = tctx.fillStyle = '#fff';
        tctx.lineWidth = stroke.size; tctx.lineCap = 'round'; tctx.lineJoin = 'round';
        tctx.beginPath();
        stroke.pts.forEach((p, i) => (i ? tctx.lineTo(p[0], p[1]) : tctx.moveTo(p[0], p[1])));
        if (stroke.pts.length === 1) { tctx.arc(stroke.pts[0][0], stroke.pts[0][1], stroke.size / 2, 0, 7); tctx.fill(); }
        else tctx.stroke();
    }
    function evToNat(e) {
        const r = brushCanvas.getBoundingClientRect();
        const { w, h } = natSize();
        const cx = (e.touches ? e.touches[0].clientX : e.clientX) - r.left;
        const cy = (e.touches ? e.touches[0].clientY : e.clientY) - r.top;
        return [Math.max(0, Math.min(w, cx / r.width * w)), Math.max(0, Math.min(h, cy / r.height * h))];
    }
    function startStroke(e) {
        if (!maskEdit.on) return;
        e.preventDefault();
        ensureMaskCanvases();
        maskEdit.drawing = true;
        maskEdit.strokes.push({ mode: maskEdit.mode, size: maskEdit.size, pts: [evToNat(e)] });
        repaintBrushPreview();
    }
    function moveStroke(e) {
        if (!maskEdit.on || !maskEdit.drawing) return;
        e.preventDefault();
        maskEdit.strokes[maskEdit.strokes.length - 1].pts.push(evToNat(e));
        repaintBrushPreview();
    }
    function endStroke() {
        if (!maskEdit.drawing) return;
        maskEdit.drawing = false;
        commitStroke(maskEdit.strokes[maskEdit.strokes.length - 1]);
    }
    brushCanvas.addEventListener('pointerdown', startStroke);
    brushCanvas.addEventListener('pointermove', moveStroke);
    window.addEventListener('pointerup', endStroke);

    function enterMaskEdit() {
        if (!cachedRoomBlob) { alert('Load or upload a room first.'); return; }
        if (visualizerState.target !== 'wall') {
            document.getElementById('target-wall-btn')?.click();
        }
        maskEdit.on = true;
        ensureMaskCanvases();
        designWrapper.classList.add('mask-editing');
        maskEditBar.style.display = 'flex';
        maskEditBtn.classList.add('active');
        repaintBrushPreview();
    }
    function exitMaskEdit() {
        maskEdit.on = false;
        designWrapper.classList.remove('mask-editing');
        maskEditBar.style.display = 'none';
        maskEditBtn.classList.remove('active');
        brushCanvas.getContext('2d').clearRect(0, 0, brushCanvas.width, brushCanvas.height);
    }
    function canvasToBlob(c) {
        return new Promise(res => c.toBlob(b => res(b), 'image/png'));
    }
    async function applyMaskEdits() {
        if (addMaskCanvas)    visualizerState.wallAddBlob    = await canvasToBlob(addMaskCanvas);
        if (removeMaskCanvas) visualizerState.wallRemoveBlob = await canvasToBlob(removeMaskCanvas);
        exitMaskEdit();
        requestServerVisualization();
    }
    function clearMaskEdits() {
        const { w, h } = natSize();
        if (addMaskCanvas) { addMaskCanvas.width = w; removeMaskCanvas.width = w; }
        maskEdit.strokes = [];
        visualizerState.wallAddBlob = null;
        visualizerState.wallRemoveBlob = null;
        repaintBrushPreview();
        requestServerVisualization();
    }
    window.resetManualWallMasks = function () {
        addMaskCanvas = removeMaskCanvas = null;
        maskEdit.strokes = [];
        visualizerState.wallAddBlob = null;
        visualizerState.wallRemoveBlob = null;
        if (maskEdit.on) exitMaskEdit();
        try { brushCanvas.getContext('2d').clearRect(0, 0, brushCanvas.width, brushCanvas.height); } catch (e) {}
    };
    if (maskEditBtn) maskEditBtn.addEventListener('click', () => maskEdit.on ? exitMaskEdit() : enterMaskEdit());
    document.querySelectorAll('.mask-edit-mode').forEach(b => b.addEventListener('click', () => {
        document.querySelectorAll('.mask-edit-mode').forEach(x => x.classList.remove('active'));
        b.classList.add('active');
        maskEdit.mode = b.dataset.mmode;
    }));
    if (brushSizeInput) brushSizeInput.addEventListener('input', e => { maskEdit.size = parseInt(e.target.value, 10); });
    document.getElementById('mask-edit-undo')?.addEventListener('click', () => {
        maskEdit.strokes.pop();
        const { w } = natSize();
        addMaskCanvas.width = w; removeMaskCanvas.width = w;         // wipe & re-commit remaining
        maskEdit.strokes.forEach(commitStroke);
        repaintBrushPreview();
    });
    document.getElementById('mask-edit-clear')?.addEventListener('click', clearMaskEdits);
    document.getElementById('mask-edit-cancel')?.addEventListener('click', exitMaskEdit);
    document.getElementById('mask-edit-apply')?.addEventListener('click', applyMaskEdits);

    // Compare: choose which side shows the original photo
    visualizerState.compareOriginalSide = 'right';
    document.querySelectorAll('.compare-side-btn').forEach(btn => {
        if (btn.dataset.side === 'right') btn.classList.add('active');
        else btn.classList.remove('active');
        btn.addEventListener('click', () => {
            document.querySelectorAll('.compare-side-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            visualizerState.compareOriginalSide = btn.dataset.side;
            updateCompareCanvases();
        });
    });

    // Canvas Hotspot clicks
    const hotspotFloor = document.getElementById('hotspot-floor');
    if (hotspotFloor) {
        hotspotFloor.addEventListener('click', () => {
            document.querySelectorAll('.panel-nav-tab').forEach(t => {
                if (t.dataset.tab === 'tab-materials') t.click();
            });
        });
    }

    // --- Custom Image Upload handlers ---
    const handleUploadedFile = async (file) => {
        if (!file) return;
        
        const reader = new FileReader();
        reader.onload = async (event) => {
            const imgDataUrl = event.target.result;
            
            showScreen('screen-workspace');
            
            const loading = document.getElementById('workspace-loading');
            if (loading) {
                document.getElementById('loading-text').textContent = 'Analyzing Custom Room...';
                loading.style.display = 'flex';
            }
            
            // Clear old state
            visualizerState.maskDataUrl = null;
            visualizerState.polygon = [];
            visualizerState.floorQuad = [];
            visualizerState.wallMaskDataUrl = null;
            visualizerState.wallPolygon = [];
            visualizerState.wallQuads = [];
            visualizerState.serverRenderedImage = null;
            imageCache.maskImage = new Image();
            imageCache.wallMaskImage = new Image();
            cachedRoomBlob = null;
            cachedFloorMaskBlob = null;
            cachedWallMaskBlob = null;
            cachedWallFgBlob = null;
            visualizerState.wallAddBlob = null;
            visualizerState.wallRemoveBlob = null;
            if (typeof resetManualWallMasks === "function") resetManualWallMasks();
            
            try {
                activeRoom = {
                    id: 'custom-room',
                    name: 'Custom Room Workspace',
                    sub: 'User-uploaded custom room configuration',
                    img: imgDataUrl,
                    cardImg: imgDataUrl,
                    baseScale: 1.0
                };
                document.getElementById('workspace-room-name').textContent = activeRoom.name;
                document.getElementById('workspace-room-sub').textContent = activeRoom.sub;

                await new Promise((resolve, reject) => {
                    imageCache.roomImage.onload = resolve;
                    imageCache.roomImage.onerror = reject;
                    imageCache.roomImage.src = imgDataUrl;
                });
                
                resizeCanvasToFit();
                renderVisualizer();
                
                // Cache room blob for /api/visualize calls
                cachedRoomBlob = file;
                
                const formData = new FormData();
                formData.append('file', file);
                
                const apiRes = await fetch('http://127.0.0.1:8000/api/segment', {
                    method: 'POST',
                    body: formData
                });
                
                if (!apiRes.ok) throw new Error('Custom segmentation failed');
                
                const data = await apiRes.json();
                
                const loadFloorMask = new Promise((resolve, reject) => {
                    imageCache.maskImage.onload = resolve;
                    imageCache.maskImage.onerror = reject;
                    imageCache.maskImage.src = data.floor_mask;
                });
                
                const loadWallMask = new Promise((resolve, reject) => {
                    imageCache.wallMaskImage.onload = resolve;
                    imageCache.wallMaskImage.onerror = reject;
                    imageCache.wallMaskImage.src = data.wall_mask;
                });
                
                await Promise.all([loadFloorMask, loadWallMask]);
                
                visualizerState.maskDataUrl    = data.floor_mask;
                visualizerState.polygon        = data.floor_polygon;
                visualizerState.floorQuad      = data.floor_quad || [];
                visualizerState.wallMaskDataUrl = data.wall_mask;
                visualizerState.wallPolygon    = data.wall_polygon;
                visualizerState.wallQuads      = data.wall_quads || [];
                visualizerState.floorArea      = data.floor_area || null;
                visualizerState.wallArea       = data.wall_area  || null;
                visualizerState.detectedObstacles = data.detected_obstacles || {};
                visualizerState.pixelsPerMeter = data.pixels_per_meter || 0;
                visualizerState.perspective    = data.perspective || null;

                // Cache mask blobs for /api/visualize
                cachedFloorMaskBlob = dataUrlToBlob(data.floor_mask);
                cachedWallMaskBlob = dataUrlToBlob(data.wall_mask);
                cachedWallFgBlob = data.wall_fg_mask ? dataUrlToBlob(data.wall_fg_mask) : null;

                updateAreaPanel();
                updateCoveragePanel();

                const activePolygon = visualizerState.target === 'floor' ? data.floor_polygon : data.wall_polygon;
                drawOutlineSVG(activePolygon, data.width, data.height);
                
                if (activeFloorTile || activeWallTile) {
                    requestServerVisualization();
                } else {
                    renderVisualizer();
                }
                
            } catch (err) {
                console.error('Custom image segmentation failed:', err);
                renderVisualizer();
            } finally {
                if (loading) {
                    document.getElementById('loading-text').textContent = 'Analyzing Room Layout...';
                    loading.style.display = 'none';
                }
            }
        };
        reader.readAsDataURL(file);
    };

    // Connect both workspace toolbar and select dashboard upload boxes
    const roomUploadInput = document.getElementById('room-upload-input');
    if (roomUploadInput) {
        roomUploadInput.addEventListener('change', (e) => {
            if (e.target.files[0]) {
                handleUploadedFile(e.target.files[0]);
            }
            e.target.value = ''; // Reset input value so it triggers change event on subsequent uploads of the same file
        });
    }
    const dashboardUploadInput = document.getElementById('dashboard-upload-input');
    if (dashboardUploadInput) {
        dashboardUploadInput.addEventListener('change', (e) => {
            if (e.target.files[0]) {
                handleUploadedFile(e.target.files[0]);
            }
            e.target.value = ''; // Reset input value so it triggers change event on subsequent uploads of the same file
        });
    }
    const headerUploadInput = document.getElementById('header-upload-input');
    if (headerUploadInput) {
        headerUploadInput.addEventListener('change', (e) => {
            if (e.target.files[0]) handleUploadedFile(e.target.files[0]);
            e.target.value = '';
        });
    }

    // --- Tab Switcher Logic ---
    document.querySelectorAll('.panel-nav-tab').forEach(tab => {
        tab.addEventListener('click', () => {
            document.querySelectorAll('.panel-nav-tab').forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            
            document.querySelectorAll('.panel-tab-content').forEach(c => c.classList.remove('active-tab'));
            const content = document.getElementById(tab.dataset.tab);
            if (content) content.classList.add('active-tab');
        });
    });

    // --- Pricing Calculator & Lead form ---
    const lengthInput = document.getElementById('calc-length');
    const widthInput = document.getElementById('calc-width');
    const resultArea = document.getElementById('result-area');
    const resultCost = document.getElementById('result-cost');
    const resultGst = document.getElementById('result-gst');
    const resultShipping = document.getElementById('result-shipping');
    const resultTotal = document.getElementById('result-total');
    
    if (lengthInput && widthInput) {
        lengthInput.addEventListener('input', updateCostCalculator);
        widthInput.addEventListener('input', updateCostCalculator);
    }

    const PATTERN_LABELS = {
        grid: 'Grid', brick: 'Brick ½', brick_third: 'Brick ⅓', vertical: 'Vertical',
        diagonal: 'Diagonal', diagonal_brick: 'Diagonal Brick', herringbone: 'Herringbone',
        chevron: 'Chevron', basketweave: 'Basketweave', windmill: 'Pinwheel', slab: 'Slab',
    };
    // Extra material to buy for cutting/wastage, by layout
    function layoutWaste(pattern, slab) {
        if (slab) return 0.08;
        if (['diagonal', 'diagonal_brick', 'herringbone', 'chevron', 'windmill', 'basketweave'].includes(pattern)) return 0.15;
        return 0.10;
    }
    function currentFloorLayoutName() {
        if (visualizerState.slab) return 'Slab / Bookmatch';
        return PATTERN_LABELS[visualizerState.pattern] || 'Grid';
    }

    function updateCostCalculator() {
        const L = parseFloat(lengthInput.value) || 0;
        const Wd = parseFloat(widthInput.value) || 0;

        const floorArea = L * Wd;
        const floorWaste = layoutWaste(visualizerState.pattern, visualizerState.slab);
        const floorPurchase = floorArea * (1 + floorWaste);

        const wallArea = 2 * (L + Wd) * 8.5 * 0.85;
        const wallWaste = layoutWaste(visualizerState.wallPattern, visualizerState.wallSlab);
        const wallPurchase = wallArea * (1 + wallWaste);

        const floorTileObj = TILE_CATALOG.find(t => t.id === visualizerState.activeFloorTileId) || activeTile;
        const wallTileObj = TILE_CATALOG.find(t => t.id === visualizerState.activeWallTileId) || activeTile;

        const floorPrice = floorTileObj ? floorTileObj.price : 75;
        const wallPrice = wallTileObj ? wallTileObj.price : 75;

        let totalPurchaseSqft = 0;
        let materialCost = 0;
        let calcAreaText = '';

        if (visualizerState.target === 'floor') {
            totalPurchaseSqft = floorPurchase;
            materialCost = floorPurchase * floorPrice;
            calcAreaText = `Floor: ${floorArea.toFixed(1)} sq.ft → buy ${Math.ceil(floorPurchase)} sq.ft`;
        } else if (visualizerState.target === 'wall') {
            totalPurchaseSqft = wallPurchase;
            materialCost = wallPurchase * wallPrice;
            calcAreaText = `Wall: ${wallArea.toFixed(1)} sq.ft → buy ${Math.ceil(wallPurchase)} sq.ft`;
        } else {
            // both surfaces
            totalPurchaseSqft = floorPurchase + wallPurchase;
            materialCost = (floorPurchase * floorPrice) + (wallPurchase * wallPrice);
            calcAreaText = `Floor (${Math.ceil(floorPurchase)} sq.ft) + Wall (${Math.ceil(wallPurchase)} sq.ft) → buy ${Math.ceil(totalPurchaseSqft)} sq.ft`;
        }

        const gst = materialCost * 0.18;
        const shipping = totalPurchaseSqft > 0 ? 1500 : 0;
        const total = materialCost + gst + shipping;

        // Tiles & Boxes
        const fMM = parseTileMM(floorTileObj && floorTileObj.specs);
        const fTileSqft = (fMM.w / 1000) * (fMM.h / 1000) * 10.7639;
        const fTilesNeeded = fTileSqft > 0 ? Math.ceil(floorPurchase / fTileSqft) : 0;
        const fBoxesNeeded = fTilesNeeded > 0 ? Math.ceil(fTilesNeeded / ((fMM.w >= 1200 || fMM.h >= 1200) ? 2 : 4)) : 0;

        const wMM = parseTileMM(wallTileObj && wallTileObj.specs);
        const wTileSqft = (wMM.w / 1000) * (wMM.h / 1000) * 10.7639;
        const wTilesNeeded = wTileSqft > 0 ? Math.ceil(wallPurchase / wTileSqft) : 0;
        const wBoxesNeeded = wTilesNeeded > 0 ? Math.ceil(wTilesNeeded / ((wMM.w >= 1200 || wMM.h >= 1200) ? 2 : 4)) : 0;

        let totalBoxes = fBoxesNeeded + wBoxesNeeded;
        let totalTiles = fTilesNeeded + wTilesNeeded;
        if (visualizerState.target === 'floor') { totalBoxes = fBoxesNeeded; totalTiles = fTilesNeeded; }
        else if (visualizerState.target === 'wall') { totalBoxes = wBoxesNeeded; totalTiles = wTilesNeeded; }

        const adhesiveBags = totalPurchaseSqft > 0 ? Math.ceil(totalPurchaseSqft / 45) : 0;
        const groutKg = totalPurchaseSqft > 0 ? Math.ceil(totalPurchaseSqft / 60) : 0;

        const layoutNote = document.getElementById('calc-layout-note');
        const tilesNote = document.getElementById('calc-tiles-note');
        const resultBoxes = document.getElementById('result-boxes');
        const resultAccessories = document.getElementById('result-accessories');

        if (layoutNote) layoutNote.textContent =
            `Floor: ${currentFloorLayoutName()} (+${Math.round(floorWaste * 100)}%) · Wall: ${(PATTERN_LABELS[visualizerState.wallPattern] || 'Grid')} (+${Math.round(wallWaste * 100)}%)`;
        if (tilesNote) tilesNote.textContent = `${totalTiles.toLocaleString()} tiles total`;

        if (resultArea) resultArea.textContent = calcAreaText;
        if (resultBoxes) resultBoxes.textContent = totalBoxes ? `${totalBoxes.toLocaleString()} Boxes (${totalTiles.toLocaleString()} tiles)` : '-- Boxes';
        if (resultAccessories) resultAccessories.textContent = totalPurchaseSqft > 0 ? `${adhesiveBags} Bags Thinset (20kg) + ${groutKg} kg Grout` : '-- Bags Adhesive + -- kg Grout';
        if (resultCost) resultCost.textContent = `₹${Math.round(materialCost).toLocaleString()}`;
        if (resultGst) resultGst.textContent = `₹${Math.round(gst).toLocaleString()}`;
        if (resultShipping) resultShipping.textContent = `₹${shipping.toLocaleString()}`;
        if (resultTotal) resultTotal.textContent = `₹${Math.round(total).toLocaleString()}`;
    }

    const btnPrintQuote = document.getElementById('btn-print-quote');
    if (btnPrintQuote) {
        btnPrintQuote.addEventListener('click', () => {
            const quoteId = 'BSY-' + new Date().getFullYear() + '-' + Math.floor(1000 + Math.random() * 9000);
            const pdfQuoteId = document.getElementById('pdf-quote-id');
            const pdfDate = document.getElementById('pdf-date');
            if (pdfQuoteId) pdfQuoteId.textContent = quoteId;
            if (pdfDate) pdfDate.textContent = new Date().toLocaleDateString('en-GB', { day: '2-digit', month: 'short', year: 'numeric' });

            const nameVal = document.getElementById('form-name')?.value || 'Valued Client';
            const phoneVal = document.getElementById('form-phone')?.value || '--';
            const emailVal = document.getElementById('form-email')?.value || '--';
            const cityVal = document.getElementById('form-city')?.value || '--';

            const pdfClientName = document.getElementById('pdf-client-name');
            const pdfClientPhone = document.getElementById('pdf-client-phone');
            const pdfClientEmail = document.getElementById('pdf-client-email');
            const pdfClientCity = document.getElementById('pdf-client-city');

            if (pdfClientName) pdfClientName.textContent = nameVal;
            if (pdfClientPhone) pdfClientPhone.textContent = phoneVal;
            if (pdfClientEmail) pdfClientEmail.textContent = emailVal;
            if (pdfClientCity) pdfClientCity.textContent = cityVal;

            const targetText = visualizerState.target === 'both' ? 'Floor & Wall Tiling (Both Surfaces)' : (visualizerState.target === 'wall' ? 'Wall Tiling Only' : 'Floor Tiling Only');
            const pdfTarget = document.getElementById('pdf-target');
            if (pdfTarget) pdfTarget.textContent = targetText;

            const L = parseFloat(lengthInput.value) || 0;
            const Wd = parseFloat(widthInput.value) || 0;

            const floorArea = L * Wd;
            const floorWaste = layoutWaste(visualizerState.pattern, visualizerState.slab);
            const floorPurchase = floorArea * (1 + floorWaste);

            const wallArea = 2 * (L + Wd) * 8.5 * 0.85;
            const wallWaste = layoutWaste(visualizerState.wallPattern, visualizerState.wallSlab);
            const wallPurchase = wallArea * (1 + wallWaste);

            const floorTileObj = TILE_CATALOG.find(t => t.id === visualizerState.activeFloorTileId) || activeTile;
            const wallTileObj = TILE_CATALOG.find(t => t.id === visualizerState.activeWallTileId) || activeTile;

            const pdfRoomDims = document.getElementById('pdf-room-dims');
            const pdfNetSqft = document.getElementById('pdf-net-sqft');
            const pdfPatternName = document.getElementById('pdf-pattern-name');

            if (pdfRoomDims) pdfRoomDims.textContent = `${L.toFixed(1)} ft × ${Wd.toFixed(1)} ft`;
            if (pdfNetSqft) pdfNetSqft.textContent = `Floor: ${floorArea.toFixed(1)} sq.ft | Wall: ${wallArea.toFixed(1)} sq.ft`;
            if (pdfPatternName) pdfPatternName.textContent = `Floor: ${currentFloorLayoutName()} (+${Math.round(floorWaste * 100)}%), Wall: ${(PATTERN_LABELS[visualizerState.wallPattern] || 'Grid')} (+${Math.round(wallWaste * 100)}%)`;

            const canvas = document.getElementById('visualizer-canvas');
            const pdfRenderedImg = document.getElementById('pdf-rendered-img');
            if (canvas && pdfRenderedImg) {
                try {
                    pdfRenderedImg.src = canvas.toDataURL('image/jpeg', 0.92);
                } catch (e) {
                    console.log('Canvas export note:', e);
                }
            }

            const pdfSwatchImg = document.getElementById('pdf-swatch-img');
            const pdfTileName = document.getElementById('pdf-tile-name');
            const pdfTileSpecs = document.getElementById('pdf-tile-specs');
            const pdfTileUnitPrice = document.getElementById('pdf-tile-unit-price');

            if (visualizerState.target === 'both' || (visualizerState.activeFloorTileId && visualizerState.activeWallTileId && floorTileObj.id !== wallTileObj.id)) {
                if (pdfSwatchImg) pdfSwatchImg.src = floorTileObj.img;
                if (pdfTileName) pdfTileName.textContent = `Floor: ${floorTileObj.name} | Wall: ${wallTileObj.name}`;
                if (pdfTileSpecs) pdfTileSpecs.textContent = `Floor: ${floorTileObj.specs} · Wall: ${wallTileObj.specs}`;
                if (pdfTileUnitPrice) pdfTileUnitPrice.textContent = `Floor: ₹${floorTileObj.price}/sq.ft · Wall: ₹${wallTileObj.price}/sq.ft`;
            } else {
                const singleTile = visualizerState.target === 'wall' ? wallTileObj : floorTileObj;
                if (pdfSwatchImg) pdfSwatchImg.src = singleTile.img;
                if (pdfTileName) pdfTileName.textContent = `${singleTile.brand} - ${singleTile.name}`;
                if (pdfTileSpecs) pdfTileSpecs.textContent = singleTile.specs;
                if (pdfTileUnitPrice) pdfTileUnitPrice.textContent = `₹${singleTile.price} / sq.ft`;
            }

            // Populate BOM table rows dynamically
            const fMM = parseTileMM(floorTileObj && floorTileObj.specs);
            const fTileSqft = (fMM.w / 1000) * (fMM.h / 1000) * 10.7639;
            const fTilesNeeded = fTileSqft > 0 ? Math.ceil(floorPurchase / fTileSqft) : 0;
            const fBoxesNeeded = fTilesNeeded > 0 ? Math.ceil(fTilesNeeded / ((fMM.w >= 1200 || fMM.h >= 1200) ? 2 : 4)) : 0;

            const wMM = parseTileMM(wallTileObj && wallTileObj.specs);
            const wTileSqft = (wMM.w / 1000) * (wMM.h / 1000) * 10.7639;
            const wTilesNeeded = wTileSqft > 0 ? Math.ceil(wallPurchase / wTileSqft) : 0;
            const wBoxesNeeded = wTilesNeeded > 0 ? Math.ceil(wTilesNeeded / ((wMM.w >= 1200 || wMM.h >= 1200) ? 2 : 4)) : 0;

            const floorCost = floorPurchase * (floorTileObj ? floorTileObj.price : 75);
            const wallCost = wallPurchase * (wallTileObj ? wallTileObj.price : 75);

            const tbody = document.querySelector('.pdf-bom-table tbody');
            let bomHtml = '';

            if (visualizerState.target === 'floor') {
                bomHtml += `
                <tr>
                    <td>1</td>
                    <td><strong>Floor Tile: ${floorTileObj.brand} - ${floorTileObj.name}</strong></td>
                    <td>${floorTileObj.specs} · ${currentFloorLayoutName()}</td>
                    <td>${Math.ceil(floorPurchase)} sq.ft (${fBoxesNeeded} Boxes / ${fTilesNeeded} Tiles)</td>
                    <td>₹${floorTileObj.price} / sq.ft</td>
                    <td style="text-align:right;">₹${Math.round(floorCost).toLocaleString()}</td>
                </tr>`;
            } else if (visualizerState.target === 'wall') {
                bomHtml += `
                <tr>
                    <td>1</td>
                    <td><strong>Wall Tile: ${wallTileObj.brand} - ${wallTileObj.name}</strong></td>
                    <td>${wallTileObj.specs} · ${(PATTERN_LABELS[visualizerState.wallPattern] || 'Grid')}</td>
                    <td>${Math.ceil(wallPurchase)} sq.ft (${wBoxesNeeded} Boxes / ${wTilesNeeded} Tiles)</td>
                    <td>₹${wallTileObj.price} / sq.ft</td>
                    <td style="text-align:right;">₹${Math.round(wallCost).toLocaleString()}</td>
                </tr>`;
            } else {
                bomHtml += `
                <tr>
                    <td>1</td>
                    <td><strong>Floor Tile: ${floorTileObj.brand} - ${floorTileObj.name}</strong></td>
                    <td>${floorTileObj.specs} · ${currentFloorLayoutName()}</td>
                    <td>${Math.ceil(floorPurchase)} sq.ft (${fBoxesNeeded} Boxes / ${fTilesNeeded} Tiles)</td>
                    <td>₹${floorTileObj.price} / sq.ft</td>
                    <td style="text-align:right;">₹${Math.round(floorCost).toLocaleString()}</td>
                </tr>
                <tr>
                    <td>2</td>
                    <td><strong>Wall Tile: ${wallTileObj.brand} - ${wallTileObj.name}</strong></td>
                    <td>${wallTileObj.specs} · ${(PATTERN_LABELS[visualizerState.wallPattern] || 'Grid')}</td>
                    <td>${Math.ceil(wallPurchase)} sq.ft (${wBoxesNeeded} Boxes / ${wTilesNeeded} Tiles)</td>
                    <td>₹${wallTileObj.price} / sq.ft</td>
                    <td style="text-align:right;">₹${Math.round(wallCost).toLocaleString()}</td>
                </tr>`;
            }

            const totalPurchaseSqft = visualizerState.target === 'floor' ? floorPurchase : (visualizerState.target === 'wall' ? wallPurchase : floorPurchase + wallPurchase);
            const totalMatCost = visualizerState.target === 'floor' ? floorCost : (visualizerState.target === 'wall' ? wallCost : floorCost + wallCost);

            const adhesiveBags = totalPurchaseSqft > 0 ? Math.ceil(totalPurchaseSqft / 45) : 0;
            const groutKg = totalPurchaseSqft > 0 ? Math.ceil(totalPurchaseSqft / 60) : 0;
            const sNoAcc = visualizerState.target === 'both' ? 3 : 2;

            bomHtml += `
            <tr>
                <td>${sNoAcc}</td>
                <td>Tile Thinset Adhesive (20kg Bags)</td>
                <td>High Bond Polymer Modified</td>
                <td>${adhesiveBags} Bags</td>
                <td>Included</td>
                <td style="text-align:right;">--</td>
            </tr>
            <tr>
                <td>${sNoAcc + 1}</td>
                <td>Epoxy / Polymer Grout (kg)</td>
                <td>Matching Joint Filler</td>
                <td>${groutKg} kg</td>
                <td>Included</td>
                <td style="text-align:right;">--</td>
            </tr>
            <tr>
                <td>${sNoAcc + 2}</td>
                <td>Freight, Transport & Handling</td>
                <td>Doorstep Logistics Delivery</td>
                <td>1 Service</td>
                <td>₹1,500</td>
                <td style="text-align:right;">₹1,500</td>
            </tr>`;

            if (tbody) tbody.innerHTML = bomHtml;

            const shipping = totalPurchaseSqft > 0 ? 1500 : 0;
            const subtotal = totalMatCost + shipping;
            const gst = totalMatCost * 0.18;
            const total = totalMatCost + gst + shipping;

            const pdfSubtotal = document.getElementById('pdf-subtotal');
            const pdfGst = document.getElementById('pdf-gst');
            const pdfGrandTotal = document.getElementById('pdf-grand-total');

            if (pdfSubtotal) pdfSubtotal.textContent = `₹${Math.round(subtotal).toLocaleString()}`;
            if (pdfGst) pdfGst.textContent = `₹${Math.round(gst).toLocaleString()}`;
            if (pdfGrandTotal) pdfGrandTotal.textContent = `₹${Math.round(total).toLocaleString()}`;

            window.print();
        });
    }

    // Fill the AI coverage panel from the last segmentation result
    function updateCoveragePanel() {
        const rows = document.getElementById('ai-coverage-rows');
        if (!rows) return;
        const fa = visualizerState.floorArea;
        const wa = visualizerState.wallArea;
        const dims = visualizerState.perspective && visualizerState.perspective.room_dims_ft;
        if (!fa && !wa && !dims) {
            rows.innerHTML = '<div class="calc-row"><span>Analyze a room first to see detected coverage.</span><span></span></div>';
            return;
        }

        let obsRatio = 0.15;
        if (fa && fa.total_sqft > 0 && fa.net_sqft >= 0 && fa.net_sqft <= fa.total_sqft) {
            obsRatio = Math.min(0.5, 1 - fa.net_sqft / fa.total_sqft);
        }
        const roomSqft = dims ? dims[0] * dims[1] : (fa ? fa.total_sqft : 0);
        const floorNet = roomSqft * (1 - obsRatio);
        const wallGross = dims ? 2 * (dims[0] + dims[1]) * 8.5 : (wa ? wa.total_sqft : 0);
        const wallNet = wallGross * 0.85;

        const floorWaste = layoutWaste(visualizerState.pattern, visualizerState.slab);
        const wallWaste = layoutWaste(visualizerState.wallPattern, visualizerState.wallSlab);

        const floorRec = Math.ceil(floorNet * (1 + floorWaste));
        const wallRec = Math.ceil(wallNet * (1 + wallWaste));
        const totalRec = floorRec + wallRec;

        let html = '';
        if (dims) html += `<div class="calc-row"><span>Detected room dimensions</span><span>${dims[0]} ft × ${dims[1]} ft</span></div>`;
        html += `<div class="calc-row"><span>Floor coverage (net of furniture)</span><span>${floorNet.toFixed(1)} sq.ft → <strong>Buy ${floorRec} sq.ft</strong> (+${Math.round(floorWaste * 100)}%)</span></div>`;
        html += `<div class="calc-row"><span>Wall coverage (net of openings)</span><span>${wallNet.toFixed(1)} sq.ft → <strong>Buy ${wallRec} sq.ft</strong> (+${Math.round(wallWaste * 100)}%)</span></div>`;
        html += `<div class="calc-row" style="border-top:1px dashed var(--border-color); padding-top:6px; font-weight:700; margin-top:4px;"><span>Total Combined Tile Purchase</span><span style="color:var(--primary); font-size:14px;">${totalRec.toLocaleString()} sq.ft</span></div>`;

        rows.innerHTML = html;
        visualizerState._calcLB = dims ? [dims[0], dims[1]] : null;
    }

    const calcUseDetected = document.getElementById('calc-use-detected');
    if (calcUseDetected) {
        calcUseDetected.addEventListener('click', () => {
            const dims = visualizerState.perspective && visualizerState.perspective.room_dims_ft;
            const fa = visualizerState.floorArea;
            if (dims) {
                lengthInput.value = dims[0];
                widthInput.value = dims[1];
            } else if (fa && fa.net_sqft) {
                const b = Math.sqrt(fa.net_sqft / 1.35);
                lengthInput.value = (fa.net_sqft / b).toFixed(1);
                widthInput.value = b.toFixed(1);
            }
            updateCostCalculator();
        });
    }

    function updateCostEstimator() {
        const container = document.getElementById('quote-materials-container');
        if (container) {
            const floorTileObj = TILE_CATALOG.find(t => t.id === visualizerState.activeFloorTileId) || activeTile;
            const wallTileObj = TILE_CATALOG.find(t => t.id === visualizerState.activeWallTileId) || activeTile;

            let html = '';
            if (visualizerState.target === 'both' || (visualizerState.activeFloorTileId && visualizerState.activeWallTileId && floorTileObj.id !== wallTileObj.id)) {
                html += `
                <div class="selected-material-card-large">
                    <img src="${floorTileObj.img}" alt="${floorTileObj.name}">
                    <div>
                        <span style="font-size:10px; font-weight:700; color:var(--primary); text-transform:uppercase;">FLOOR TILE MATERIAL</span>
                        <h3 style="font-size:15px; font-weight:700; margin:2px 0;">${floorTileObj.brand} - ${floorTileObj.name}</h3>
                        <p style="font-size:12px; color:var(--text-muted); margin: 2px 0 4px;">${floorTileObj.specs} | Layout: ${currentFloorLayoutName()}</p>
                        <span style="font-size:14px; font-weight:700; color: var(--primary);">₹${floorTileObj.price} / sq.ft</span>
                    </div>
                </div>
                <div class="selected-material-card-large">
                    <img src="${wallTileObj.img}" alt="${wallTileObj.name}">
                    <div>
                        <span style="font-size:10px; font-weight:700; color:var(--primary); text-transform:uppercase;">WALL TILE MATERIAL</span>
                        <h3 style="font-size:15px; font-weight:700; margin:2px 0;">${wallTileObj.brand} - ${wallTileObj.name}</h3>
                        <p style="font-size:12px; color:var(--text-muted); margin: 2px 0 4px;">${wallTileObj.specs} | Layout: ${(PATTERN_LABELS[visualizerState.wallPattern] || 'Grid')}</p>
                        <span style="font-size:14px; font-weight:700; color: var(--primary);">₹${wallTileObj.price} / sq.ft</span>
                    </div>
                </div>`;
            } else {
                const singleTile = visualizerState.target === 'wall' ? wallTileObj : floorTileObj;
                html += `
                <div class="selected-material-card-large">
                    <img src="${singleTile.img}" alt="${singleTile.name}">
                    <div>
                        <span style="font-size:10px; font-weight:700; color:var(--primary); text-transform:uppercase;">${visualizerState.target.toUpperCase()} TILE MATERIAL</span>
                        <h3 style="font-size:15px; font-weight:700; margin:2px 0;">${singleTile.brand} - ${singleTile.name}</h3>
                        <p style="font-size:12px; color:var(--text-muted); margin: 2px 0 4px;">${singleTile.specs}</p>
                        <span style="font-size:14px; font-weight:700; color: var(--primary);">₹${singleTile.price} / sq.ft</span>
                    </div>
                </div>`;
            }
            container.innerHTML = html;
        }

        updateCoveragePanel();
        updateCostCalculator();
    }
    
    document.getElementById('workspace-cost-btn').addEventListener('click', () => {
        showScreen('screen-quote');
    });
    document.getElementById('workspace-quote-btn').addEventListener('click', () => {
        showScreen('screen-quote');
    });
    document.getElementById('quote-back-btn').addEventListener('click', () => {
        showScreen('screen-workspace');
    });
    
    // Lead request form submission logic
    const leadForm = document.getElementById('quote-lead-form');
    const submitBtn = document.getElementById('submit-quote-btn');
    if (leadForm) {
        leadForm.addEventListener('submit', (e) => {
            e.preventDefault();
            
            if (submitBtn) {
                submitBtn.disabled = true;
                const originalText = submitBtn.innerHTML;
                submitBtn.innerHTML = `
                    <svg class="animate-spin" style="animation: spin 1s linear infinite; width: 14px; height: 14px; margin-right: 8px; fill: none; stroke: currentColor; stroke-width: 2.5;" viewBox="0 0 24 24">
                        <circle cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" style="opacity: 0.25;"></circle>
                        <path fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    Generating Estimate...
                `;
                
                if (!document.getElementById('spin-keyframes')) {
                    const style = document.createElement('style');
                    style.id = 'spin-keyframes';
                    style.innerHTML = `@keyframes spin { 100% { transform: rotate(360deg); } }`;
                    document.head.appendChild(style);
                }
                
                setTimeout(() => {
                    submitBtn.disabled = false;
                    submitBtn.innerHTML = originalText;
                    
                    const refId = `REF-VR-2026-${Math.floor(1000 + Math.random() * 9000)}`;
                    document.getElementById('ref-id-display').textContent = refId;
                    
                    leadForm.reset();
                    showScreen('screen-success');
                }, 1500);
            }
        });
    }

    // Copy Ref ID
    const copyRefBtn = document.getElementById('copy-ref-btn');
    if (copyRefBtn) {
        copyRefBtn.addEventListener('click', () => {
            const refText = document.getElementById('ref-id-display').textContent;
            navigator.clipboard.writeText(refText).then(() => {
                const originalSVG = copyRefBtn.innerHTML;
                copyRefBtn.innerHTML = `
                    <svg viewBox="0 0 24 24" style="stroke: var(--accent); width:14px; height:14px; fill:none;"><path d="M5 13l4 4L19 7" stroke-width="3.5" stroke-linecap="round" stroke-linejoin="round"/></svg>
                `;
                setTimeout(() => {
                    copyRefBtn.innerHTML = originalSVG;
                }, 1500);
            });
        });
    }

    // Navigation switches
    document.getElementById('success-home-btn').addEventListener('click', () => {
        showScreen('screen-hero');
    });
    document.getElementById('success-visualizer-btn').addEventListener('click', () => {
        showScreen('screen-workspace');
    });
    document.getElementById('logo-btn').addEventListener('click', (e) => {
        e.preventDefault();
        showScreen('screen-hero');
    });
    document.getElementById('nav-home').addEventListener('click', (e) => {
        e.preventDefault();
        showScreen('screen-hero');
    });
    document.getElementById('nav-visualizer').addEventListener('click', (e) => {
        e.preventDefault();
        showScreen('screen-dashboard'); // Landing opens Select Dashboard
    });
    document.getElementById('nav-calculator').addEventListener('click', (e) => {
        e.preventDefault();
        showScreen('screen-quote');
    });
    document.getElementById('hero-start-btn').addEventListener('click', () => {
        showScreen('screen-dashboard');
    });
    document.getElementById('header-cta-btn').addEventListener('click', () => {
        showScreen('screen-dashboard');
    });
    document.getElementById('hero-learn-btn').addEventListener('click', () => {
        const featuresSection = document.querySelector('.features-grid');
        if (featuresSection) {
            featuresSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
    });

    window.addEventListener('resize', () => {
        if (currentScreen === 'screen-workspace') {
            resizeCanvasToFit();
            renderVisualizer();
        }
    });

    // Initializations
    initDynamicUI();
    selectTile(materialsData[0]);
    loadRoom(roomsData[0]);
});
