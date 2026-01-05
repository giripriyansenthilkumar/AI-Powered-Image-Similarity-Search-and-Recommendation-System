// Fashion Image Similarity Frontend Logic
// Enhanced with accessibility, error handling, and improved UX

const API_URL = 'http://localhost:8000/api/search';
const API_ORIGIN = new URL(API_URL).origin;
const imageInput = document.getElementById('imageInput');
const previewPanel = document.getElementById('previewPanel');
const previewImage = document.getElementById('previewImage');
const removeImageBtn = document.getElementById('removeImageBtn');
const findSimilarBtn = document.getElementById('findSimilarBtn');
const statusMessage = document.getElementById('statusMessage');
const loader = document.getElementById('loader');
const resultsGrid = document.getElementById('resultsGrid');
const uploadCard = document.querySelector('.upload-card');
const cameraPreview = document.getElementById('cameraPreview');
const startCameraBtn = document.getElementById('startCameraBtn');
const capturePhotoBtn = document.getElementById('capturePhotoBtn');
const stopCameraBtn = document.getElementById('stopCameraBtn');

let selectedFile = null;
let isLoading = false;
let cameraStream = null;

// Drag & drop support with improved accessibility
uploadCard.addEventListener('dragover', (e) => {
  e.preventDefault();
  e.stopPropagation();
  uploadCard.classList.add('dragover');
  uploadCard.setAttribute('aria-pressed', 'true');
});

uploadCard.addEventListener('dragleave', (e) => {
  e.preventDefault();
  e.stopPropagation();
  uploadCard.classList.remove('dragover');
  uploadCard.setAttribute('aria-pressed', 'false');
});

uploadCard.addEventListener('drop', (e) => {
  e.preventDefault();
  e.stopPropagation();
  uploadCard.classList.remove('dragover');
  uploadCard.setAttribute('aria-pressed', 'false');
  if (e.dataTransfer.files && e.dataTransfer.files[0]) {
    handleFile(e.dataTransfer.files[0]);
  }
});

// Camera helpers
async function startCamera() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    showStatus('Camera is not supported in this browser.', true);
    return;
  }

  if (cameraStream) {
    showStatus('Camera already running. Capture when ready.');
    return;
  }

  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } });
    if (cameraPreview) {
      cameraPreview.srcObject = cameraStream;
      cameraPreview.hidden = false;
      await cameraPreview.play().catch(() => {});
    }
    if (capturePhotoBtn) capturePhotoBtn.disabled = false;
    if (stopCameraBtn) stopCameraBtn.disabled = false;
    if (startCameraBtn) startCameraBtn.disabled = true;
    showStatus('Camera ready. Tap Capture to take a photo.');
  } catch (err) {
    showStatus('Camera access blocked. Please allow permission and try again.', true);
    console.error('Camera error:', err);
  }
}

function stopCamera() {
  if (cameraStream) {
    cameraStream.getTracks().forEach(track => track.stop());
    cameraStream = null;
  }
  if (cameraPreview) {
    cameraPreview.srcObject = null;
    cameraPreview.hidden = true;
  }
  if (capturePhotoBtn) capturePhotoBtn.disabled = true;
  if (stopCameraBtn) stopCameraBtn.disabled = true;
  if (startCameraBtn) startCameraBtn.disabled = false;
}

function capturePhoto() {
  if (!cameraStream || !cameraPreview || cameraPreview.videoWidth === 0) {
    showStatus('Camera not ready yet. Please wait a second.', true);
    return;
  }

  const canvas = document.createElement('canvas');
  canvas.width = cameraPreview.videoWidth;
  canvas.height = cameraPreview.videoHeight;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(cameraPreview, 0, 0, canvas.width, canvas.height);

  canvas.toBlob((blob) => {
    if (!blob) {
      showStatus('Could not capture image. Please try again.', true);
      return;
    }
    const file = new File([blob], `camera-capture-${Date.now()}.png`, { type: 'image/png' });
    handleFile(file);
    showStatus('Captured from camera. Ready to search.');
  }, 'image/png', 0.92);
}

if (startCameraBtn) startCameraBtn.addEventListener('click', startCamera);
if (capturePhotoBtn) capturePhotoBtn.addEventListener('click', capturePhoto);
if (stopCameraBtn) stopCameraBtn.addEventListener('click', stopCamera);

// File input change
imageInput.addEventListener('change', (e) => {
  if (e.target.files && e.target.files[0]) {
    handleFile(e.target.files[0]);
  }
});

// Handle file selection/validation with enhanced error messaging
function handleFile(file) {
  // Validate file type
  const validImageTypes = ['image/jpeg', 'image/png', 'image/gif', 'image/webp', 'image/svg+xml'];
  if (!validImageTypes.includes(file.type)) {
    showStatus('Invalid file format. Please upload JPG, PNG, GIF, WebP, or SVG.', true);
    imageInput.value = '';
    return;
  }
  
  // Validate file size (5MB limit)
  const maxSize = 5 * 1024 * 1024;
  if (file.size > maxSize) {
    showStatus(`File too large (${(file.size / 1024 / 1024).toFixed(1)}MB). Maximum is 5MB.`, true);
    imageInput.value = '';
    return;
  }
  
  // Validate minimum file size (10KB)
  const minSize = 10 * 1024;
  if (file.size < minSize) {
    showStatus('File is too small. Please select a larger image.', true);
    imageInput.value = '';
    return;
  }
  
  selectedFile = file;
  const reader = new FileReader();
  
  reader.onload = function(e) {
    previewImage.src = e.target.result;
    previewImage.alt = `Preview of ${file.name}`;
    previewPanel.hidden = false;
    findSimilarBtn.disabled = false;
    showStatus(`Selected: ${file.name} (${(file.size / 1024).toFixed(1)}KB)`);
    uploadCard.setAttribute('aria-pressed', 'true');
  };
  
  reader.onerror = function() {
    showStatus('Failed to read file. Please try again.', true);
    imageInput.value = '';
  };
  
  reader.readAsDataURL(file);
}

// Remove image
removeImageBtn.addEventListener('click', () => {
  resetUpload();
});

function resetUpload() {
  selectedFile = null;
  imageInput.value = '';
  previewPanel.hidden = true;
  previewImage.src = 'assets/placeholder.svg';
  previewImage.alt = 'Image preview placeholder';
  findSimilarBtn.disabled = true;
  showStatus('');
  uploadCard.setAttribute('aria-pressed', 'false');
  stopCamera();
}

// Find Similar button with improved error handling
findSimilarBtn.addEventListener('click', async () => {
  if (!selectedFile) {
    showStatus('No image selected.', true);
    return;
  }
  
  if (isLoading) {
    return;
  }
  
  isLoading = true;
  findSimilarBtn.disabled = true;
  showLoader(true);
  showStatus('Searching for similar items...');
  resultsGrid.innerHTML = '';
  resultsGrid.setAttribute('aria-busy', 'true');
  
  try {
    const formData = new FormData();
    formData.append('image', selectedFile);

    // Set timeout for request (10 seconds)
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 10000);

    const response = await fetch(API_URL, {
      method: 'POST',
      body: formData,
      signal: controller.signal,
    });

    clearTimeout(timeoutId);

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `Server error (${response.status}). Please try again.`);
    }

    const data = await response.json();
    
    if (!data.results || !Array.isArray(data.results)) {
      throw new Error('Invalid response format from server.');
    }
    
    if (data.results.length === 0) {
      showStatus('No similar images found. Try uploading a different image.', false);
      showLoader(false);
      resultsGrid.setAttribute('aria-busy', 'false');
      return;
    }
    
    renderResults(data.results);
    showStatus(`Found ${data.results.length} similar items!`);
    resultsGrid.setAttribute('aria-busy', 'false');
    
    // Announce to screen readers
    const announcement = document.createElement('div');
    announcement.setAttribute('role', 'status');
    announcement.setAttribute('aria-live', 'assertive');
    announcement.textContent = `Results loaded. ${data.results.length} similar fashion items found.`;
    announcement.style.position = 'absolute';
    announcement.style.left = '-10000px';
    document.body.appendChild(announcement);
    setTimeout(() => announcement.remove(), 1000);
    
  } catch (err) {
    let errorMsg = 'An error occurred.';
    
    if (err.name === 'AbortError') {
      errorMsg = 'Request timeout. Backend may be unavailable.';
    } else if (err instanceof TypeError) {
      errorMsg = 'Connection error. Make sure the backend is running.';
    } else {
      errorMsg = err.message || errorMsg;
    }
    
    showStatus(errorMsg, true);
    resultsGrid.setAttribute('aria-busy', 'false');
    console.error('Search error:', err);
  } finally {
    showLoader(false);
    isLoading = false;
    findSimilarBtn.disabled = false;
  }
});

// Render results with accessibility enhancements
function renderResults(results) {
  resultsGrid.innerHTML = '';
  results.forEach((item, index) => {
    const card = document.createElement('div');
    card.className = 'result-card';
    card.tabIndex = 0;
    card.setAttribute('role', 'article');
    card.setAttribute('aria-label', `Similar item ${index + 1} with ${(item.similarity * 100).toFixed(1)}% similarity`);
    
    const img = document.createElement('img');
    img.alt = `Similar fashion item - Similarity: ${(item.similarity * 100).toFixed(1)}%`;
    img.loading = 'lazy';
    
    // Fallback to placeholder if image fails to load
    const imgUrl = item.image_url && item.image_url.startsWith('/') ? `${API_ORIGIN}${item.image_url}` : item.image_url;
    img.src = imgUrl;
    img.onerror = function() {
      // Use a placeholder with the index as seed
      this.src = `https://via.placeholder.com/400/333333/999999?text=Fashion+Item+${item.index}`;
      this.alt += ' (placeholder - dataset not available)';
      card.classList.add('placeholder-image');
    };
    
    const scoreDiv = document.createElement('div');
    scoreDiv.className = 'similarity-score';
    scoreDiv.setAttribute('aria-label', 'Similarity score');
    scoreDiv.textContent = `Similarity: ${(item.similarity * 100).toFixed(1)}%`;
    
    card.appendChild(img);
    card.appendChild(scoreDiv);
    
    // Add keyboard interaction
    card.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        // Could add more interaction here (e.g., open in modal)
      }
    });
    
    resultsGrid.appendChild(card);
  });
}

// Loader and status with improved accessibility
function showLoader(show) {
  loader.hidden = !show;
  if (show) {
    loader.setAttribute('aria-busy', 'true');
  } else {
    loader.setAttribute('aria-busy', 'false');
  }
}

function showStatus(msg, isError = false) {
  statusMessage.textContent = msg;
  statusMessage.style.color = isError ? '#ff6f91' : '#20c997';
  statusMessage.setAttribute('role', isError ? 'alert' : 'status');
  
  // Announce to screen readers
  if (msg) {
    statusMessage.setAttribute('aria-live', isError ? 'assertive' : 'polite');
    statusMessage.setAttribute('aria-atomic', 'true');
  }
}

// Accessibility: keyboard navigation for upload card
uploadCard.addEventListener('keydown', (e) => {
  if ((e.key === 'Enter' || e.key === ' ') && document.activeElement === uploadCard) {
    e.preventDefault();
    imageInput.click();
  }
});

// Keyboard shortcut help
document.addEventListener('keydown', (e) => {
  // Shift+U to focus upload input
  if (e.shiftKey && e.key === 'U') {
    e.preventDefault();
    imageInput.click();
  }
  // Shift+F to find similar
  if (e.shiftKey && e.key === 'F' && !findSimilarBtn.disabled) {
    e.preventDefault();
    findSimilarBtn.click();
  }
});

// Topbar navigation logic with ARIA attributes
const navDashboard = document.getElementById('navDashboard');
const navAbout = document.getElementById('navAbout');
const navHelp = document.getElementById('navHelp');
const dashboardSection = document.getElementById('dashboardSection');
const aboutSection = document.getElementById('aboutSection');
const helpSection = document.getElementById('helpSection');

function showSection(section) {
  // Hide all sections
  if (dashboardSection) dashboardSection.classList.add('hidden-section');
  if (aboutSection) aboutSection.classList.add('hidden-section');
  if (helpSection) helpSection.classList.add('hidden-section');
  
  // Show selected section
  const sections = {
    dashboard: dashboardSection,
    about: aboutSection,
    help: helpSection
  };
  
  if (sections[section]) {
    sections[section].classList.remove('hidden-section');
  }
  
  // Update nav buttons
  const navButtons = [navDashboard, navAbout, navHelp];
  navButtons.forEach(nav => {
    if (nav) {
      nav.classList.remove('active');
      nav.setAttribute('aria-selected', 'false');
      nav.setAttribute('tabindex', '-1');
    }
  });
  
  const navMap = { dashboard: navDashboard, about: navAbout, help: navHelp };
  const activeNav = navMap[section];
  if (activeNav) {
    activeNav.classList.add('active');
    activeNav.setAttribute('aria-selected', 'true');
    activeNav.setAttribute('tabindex', '0');
    activeNav.focus();
  }
}

if (navDashboard) navDashboard.addEventListener('click', (e) => { e.preventDefault(); showSection('dashboard'); });
if (navAbout) navAbout.addEventListener('click', (e) => { e.preventDefault(); showSection('about'); });
if (navHelp) navHelp.addEventListener('click', (e) => { e.preventDefault(); showSection('help'); });

// Tab keyboard navigation
const tablist = document.querySelector('[role="tablist"]');
if (tablist) {
  const tabs = tablist.querySelectorAll('[role="tab"]');
  
  tabs.forEach((tab, index) => {
    // Set initial tabindex
    if (tab.getAttribute('aria-selected') === 'true') {
      tab.setAttribute('tabindex', '0');
    } else {
      tab.setAttribute('tabindex', '-1');
    }
    
    tab.addEventListener('keydown', (e) => {
      let newIndex = index;
      let handled = false;
      
      if (e.key === 'ArrowRight' || e.key === 'ArrowDown') {
        e.preventDefault();
        newIndex = (index + 1) % tabs.length;
        handled = true;
      } else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') {
        e.preventDefault();
        newIndex = (index - 1 + tabs.length) % tabs.length;
        handled = true;
      } else if (e.key === 'Home') {
        e.preventDefault();
        newIndex = 0;
        handled = true;
      } else if (e.key === 'End') {
        e.preventDefault();
        newIndex = tabs.length - 1;
        handled = true;
      }
      
      if (handled) {
        tabs[newIndex].focus();
        tabs[newIndex].click();
      }
    });
  });
}

// Show dashboard by default
showSection('dashboard');

// Team member modal logic
const memberModal = document.getElementById('memberModal');
const memberModalContent = document.getElementById('memberModalContent');
const closeMemberModal = document.getElementById('closeMemberModal');

// Ensure modal is hidden on load
if (memberModal) {
  memberModal.hidden = true;
  memberModal.style.display = 'none';
}

const teamMembers = {
  giripriyan: {
    name: 'Giripriyan S',
    photo: 'https://avatars.githubusercontent.com/u/161918930?v=4',
    bio: 'Full-stack developer and AI enthusiast. Passionate about building intelligent systems that solve real-world problems. Expertise in ML pipeline development and frontend engineering.',
    profiles: [
      { label: 'GitHub', url: 'https://github.com/giripriyansenthilkumar', icon: '🐙' },
      { label: 'LinkedIn', url: 'https://linkedin.com/in/giripriyan-s', icon: '💼' }
    ]
  },
  akarshana: {
    name: 'Akarshana S',
    photo: 'https://avatars.githubusercontent.com/u/185575385?v=4',
    bio: 'Backend engineer and data specialist. Focused on building scalable systems and optimizing data pipelines. Experienced in cloud infrastructure and database design.',
    profiles: [
      { label: 'GitHub', url: 'https://github.com/akarshanas24', icon: '🐙' },
      { label: 'LinkedIn', url: 'https://www.linkedin.com/in/akarshanas/', icon: '💼' }
    ]
  },
  sivabakkiyan: {
    name: 'Sivabakkiyan I',
    photo: 'https://avatars.githubusercontent.com/u/173354924?v=4',
    bio: 'AI/ML specialist and computer vision expert. Dedicated to advancing visual search and recommendation systems. Passionate about fashion technology and sustainable AI practices.',
    profiles: [
      { label: 'GitHub', url: 'https://github.com/Sivabakkiyan', icon: '🐙' },
      { label: 'LinkedIn', url: 'https://linkedin.com/in/sivabakkiyan', icon: '💼' }
    ]
  }
};

function openMemberModal(memberKey) {
  const m = teamMembers[memberKey];
  if (!m || !memberModal) return;
  
  const profilesHTML = m.profiles.map(p => `
    <a href="${p.url}" target="_blank" rel="noopener" style="display:inline-flex;align-items:center;gap:0.6rem;background:#5f3dc4;color:#fff;padding:0.7rem 1.5rem;border-radius:0.9rem;text-decoration:none;font-weight:600;transition:background 0.18s;font-size:1rem;margin:0.5rem;">
      <span style="font-size:1.3rem;">${p.icon}</span>
      ${p.label}
    </a>
  `).join('');
  
  memberModalContent.innerHTML = `
    <div style="text-align:center;">
      <img src="${m.photo}" alt="${m.name} photo" style="width:96px;height:96px;border-radius:50%;object-fit:cover;margin-bottom:1rem;border:3px solid #5f3dc4;">
      <h2 style="margin:0.5rem 0 0.3rem 0;">${m.name}</h2>
      <div style="color:#888;font-size:1rem;margin-bottom:0.7rem;">ID: ${m.id}</div>
      <p style="margin:1rem 0 1.5rem 0;font-size:1.05rem;line-height:1.6;">${m.bio}</p>
      <div style="margin-top:1.5rem;padding-top:1.5rem;border-top:2px solid #e9ecef;">
        <div style="font-size:0.95rem;color:#666;margin-bottom:1rem;font-weight:600;">Connect On</div>
        <div style="display:flex;flex-wrap:wrap;justify-content:center;gap:0.5rem;">
          ${profilesHTML}
        </div>
      </div>
    </div>
  `;
  // Show modal centered using flex; lock background scroll for better UX
  memberModal.hidden = false;
  memberModal.style.display = 'flex';
  document.body.style.overflow = 'hidden';
  closeMemberModal && closeMemberModal.focus();
}

function closeMember() {
  if (memberModal) {
    memberModal.hidden = true;
    memberModal.style.display = 'none';
    // restore scrolling
    document.body.style.overflow = '';
  }
}

// Add click, keyboard, and hover interactions to team cards (click or hover opens modal)
const cards = document.querySelectorAll('.team-card');
let hoverOpenTimeout = null;
let hoverCloseTimeout = null;

cards.forEach(card => {
  const memberKey = card.dataset.member;
  const photo = card.querySelector('.team-photo');

  // Click opens modal
  card.addEventListener('click', (e) => {
    if (memberKey) openMemberModal(memberKey);
  });

  // Keyboard (Enter/Space) on card
  card.addEventListener('keydown', (e) => {
    if ((e.key === 'Enter' || e.key === ' ') && memberKey) {
      e.preventDefault();
      openMemberModal(memberKey);
    }
  });

  // Hover open with a short delay to avoid accidental popups
  card.addEventListener('mouseenter', () => {
    clearTimeout(hoverCloseTimeout);
    hoverOpenTimeout = setTimeout(() => {
      if (memberKey) openMemberModal(memberKey);
    }, 250);
  });

  // Cancel open or schedule close
  card.addEventListener('mouseleave', () => {
    clearTimeout(hoverOpenTimeout);
    hoverCloseTimeout = setTimeout(() => {
      if (memberModal && !memberModal.hidden) closeMember();
    }, 300);
  });

  // Keep photo keyboard behavior for accessibility
  if (photo) {
    photo.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        if (memberKey) openMemberModal(memberKey);
      }
    });
  }
});

// Keep modal open if the mouse moves into it; close when leaving the modal area
if (memberModal) {
  memberModal.addEventListener('mouseenter', () => {
    clearTimeout(hoverCloseTimeout);
  });
  memberModal.addEventListener('mouseleave', () => {
    hoverCloseTimeout = setTimeout(() => closeMember(), 300);
  });
}

if (closeMemberModal) {
  closeMemberModal.addEventListener('click', (e) => {
    e.preventDefault();
    e.stopPropagation();
    closeMember();
  });
}

window.addEventListener('keydown', (e) => {
  if (memberModal && !memberModal.hidden && (e.key === 'Escape' || e.key === 'Esc')) {
    closeMember();
  }
});

if (memberModal) {
  memberModal.addEventListener('click', (e) => {
    if (e.target === memberModal) closeMember();
  });
}
