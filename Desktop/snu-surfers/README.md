# SNU Surfer 🧭

### Smart Campus Navigation System

A Python-powered campus information, navigation, and gamified exploration platform for discovering and navigating between campus buildings and important campus spots.

---

## 📌 Overview

**SNU Surfer** is a smart campus navigation platform designed to make discovering campus locations easier, faster, and more engaging.

The system provides a centralized database of **campus buildings and important campus spots**, allowing users to:

* 🔍 Search for campus locations
* 📍 View location information
* 🗺️ Explore locations through an interactive map
* 🧭 Navigate between campus destinations
* 🎮 Earn rewards through gamified campus exploration

The core system follows:

```text
CAMPUS LOCATION DATA
        ↓
     SEARCH
        ↓
  LOCATION INFO
        ↓
   INTERACTIVE MAP
        ↓
    NAVIGATION
        ↓
   GAMIFICATION
```

---

## 🎯 Problem Statement

Large university campuses contain numerous buildings, facilities, and important locations.

Students, visitors, and staff may have difficulty:

* Finding unfamiliar locations
* Discovering important campus spots
* Quickly accessing location information
* Understanding how to get from one campus destination to another

Traditional maps may provide geographic information but do not necessarily provide a **searchable, interactive, and engaging campus experience**.

SNU Surfer addresses this by combining:

> **Campus Information + Interactive Mapping + Navigation + Gamification**

---

## 💡 Proposed Solution

SNU Surfer provides a single platform for campus discovery and navigation.

Users can:

1. Search for a building or campus spot
2. View information about the selected location
3. Locate it on an interactive map
4. Select a destination
5. Calculate a route
6. Visualize the route
7. Earn points or rewards through supported gamified activities

---

# 🗺️ Core System

The foundation of SNU Surfer is a structured **campus location database**.

## Buildings

Each building can contain:

| Field       | Description          |
| ----------- | -------------------- |
| Name        | Building name        |
| Description | Short description    |
| Category    | Type of building     |
| Latitude    | Geographic latitude  |
| Longitude   | Geographic longitude |

## Campus Spots

Each campus spot can contain:

| Field       | Description          |
| ----------- | -------------------- |
| Name        | Spot name            |
| Description | Short description    |
| Category    | Location category    |
| Latitude    | Geographic latitude  |
| Longitude   | Geographic longitude |

### Example Categories

* Academic
* Food
* Sports
* Accommodation
* Administration
* Entrance
* Parking
* Other

> Actual campus locations will be added from verified campus data. Placeholder values such as `[BUILDING NAME]` are used until the dataset is finalized.

---

# 🎮 Gamification

Gamification is designed to make campus exploration more engaging rather than simply providing directions.

Potential gamification features include:

### 🏆 Points

Users can earn points for activities such as:

* Discovering locations
* Completing navigation challenges
* Visiting designated campus spots
* Participating in campus exploration activities

### 🎖️ Badges

Users can unlock badges based on achievements.

Examples:

```text
FIRST DISCOVERY
↓
LOCATION EXPLORER
↓
CAMPUS SURFER
↓
CAMPUS MASTER
```

### 🔥 Streaks

Encourage continued exploration through activity streaks.

```text
DAY 1 → DAY 2 → DAY 3 → DAY 4
       🔥 4 DAY STREAK
```

### 🗺️ Exploration Challenges

The system can provide location-based challenges such as:

```text
CHALLENGE

Find:
[BUILDING / CAMPUS SPOT]

Navigate there
       ↓
Reach destination
       ↓
Complete challenge
       ↓
Earn XP
```

### 🏅 Leaderboards

Users can compete through campus exploration points.

Possible leaderboard metrics:

* Total XP
* Locations discovered
* Challenges completed
* Current streak

Gamification features will be implemented progressively based on the finalized requirements.

---

# ✨ Key Features

## 🔍 Location Search

Search for buildings and campus spots using names or categories.

## 🏢 Building Information

View structured information about campus buildings.

## 📍 Campus Spot Information

Discover important campus locations.

## 🗺️ Interactive Map

Explore campus locations visually using an interactive map.

## 🧭 Route Navigation

Generate routes between campus destinations.

## 🚶 Route Visualization

Display the calculated route directly on the map.

## 🎮 Gamified Exploration

Earn points, badges, streaks, and other rewards through supported campus activities.

## 📱 Responsive Interface

Provide a clean experience across different screen sizes.

## ⚡ Python Backend

Handle application logic and APIs through Flask.

---

# 🏗️ System Architecture

```text
                         USER
                           │
                           ↓
                ┌────────────────────┐
                │      FRONTEND      │
                │                    │
                │ HTML / CSS / JS    │
                │ Bootstrap          │
                │ Leaflet.js         │
                └─────────┬──────────┘
                          │
                       REST API
                          │
                          ↓
                ┌────────────────────┐
                │   PYTHON + FLASK   │
                │      BACKEND       │
                └─────────┬──────────┘
                          │
              ┌───────────┼────────────┐
              ↓           ↓            ↓
       ┌────────────┐ ┌──────────┐ ┌──────────────┐
       │   CAMPUS   │ │ NAVIGATION│ │ GAMIFICATION │
       │  LOCATION  │ │   LOGIC   │ │    LOGIC     │
       │  DATABASE  │ │           │ │              │
       │            │ │  Route    │ │ XP / Badges  │
       │ Buildings  │ │ Calculation│ │ Challenges  │
       │ Campus     │ │           │ │ Leaderboard │
       │ Spots      │ │           │ │ Streaks     │
       └──────┬─────┘ └─────┬─────┘ └──────┬───────┘
              │             │              │
              └─────────────┼──────────────┘
                            ↓
                     INTERACTIVE MAP
                            │
                    Leaflet.js + OSM
```

---

# 🛠️ Technology Stack

## Frontend

* HTML
* CSS
* JavaScript
* Bootstrap
* Leaflet.js

## Backend

* Python
* Flask
* REST APIs

## Database

* SQLite

## Map Data

* OpenStreetMap

## Navigation

* Python-based pathfinding/routing logic
* `[ROUTING ALGORITHM TO BE CONFIRMED]`

## DevOps

* Git
* GitHub
* GitHub Actions
* pytest
* Docker

### Future Scalability

PostgreSQL may be considered for larger deployments in the future.

---

# 🗺️ Mapping Architecture

### OpenStreetMap

OpenStreetMap provides the underlying geographic map data used by the application.

### Leaflet.js

Leaflet.js provides the interactive map interface.

It handles:

* Map rendering
* Markers
* Zoom
* Pan
* Location visualization
* Route visualization

Leaflet.js is **not** the routing algorithm.

The navigation flow is:

```text
Campus Database
      ↓
Building / Spot Coordinates
      ↓
Python + Flask
      ↓
Navigation Logic
      ↓
Leaflet.js
      ↓
OpenStreetMap-based Map
```

---

# 🔄 DevOps Workflow

SNU Surfer follows a development workflow designed around version control, automated testing, and containerization.

```text
PLAN
  ↓
CODE
  ↓
GIT
  ↓
GITHUB
  ↓
BUILD
  ↓
TEST
  ↓
DOCKER
  ↓
DEPLOY
  ↓
IMPROVE
```

## Git

Used for version control and tracking development changes.

## GitHub

Used for:

* Repository hosting
* Collaboration
* Pull requests
* Version management

## GitHub Actions

Automates CI/CD workflows.

Possible pipeline:

```text
Git Push
   ↓
GitHub Actions
   ↓
Install Dependencies
   ↓
Run pytest
   ↓
Build Application
   ↓
Build Docker Image
   ↓
Deploy
```

If testing fails:

```text
TEST FAIL
   ↓
PIPELINE STOPS
   ↓
DEVELOPER FIXES CODE
   ↓
GIT PUSH
   ↓
PIPELINE RUNS AGAIN
```

## pytest

Used for automated testing of the Python backend and application logic.

## Docker

Used to containerize the application and maintain consistent development and deployment environments.

---

# 📁 Suggested Project Structure

```text
snu-surfer/
│
├── app/
│   ├── __init__.py
│   ├── routes/
│   ├── services/
│   ├── models/
│   └── utils/
│
├── frontend/
│   ├── index.html
│   ├── css/
│   └── js/
│
├── data/
│   └── campus.db
│
├── tests/
│   ├── test_locations.py
│   ├── test_navigation.py
│   └── test_gamification.py
│
├── Dockerfile
├── requirements.txt
├── pytest.ini
├── README.md
└── .github/
    └── workflows/
        └── ci.yml
```

The exact structure may change as development progresses.

---

# 🔌 API Design

The Flask backend will expose REST APIs for communication between the frontend and backend.

Potential endpoints include:

```text
GET    /api/locations
GET    /api/locations/<id>
GET    /api/search
GET    /api/categories
POST   /api/navigation
GET    /api/challenges
POST   /api/challenges/<id>/complete
GET    /api/leaderboard
```

The final API structure will be determined during implementation.

---

# 🎮 Gamification Flow

A potential gamified user flow:

```text
USER
 ↓
SEARCH LOCATION
 ↓
VIEW LOCATION
 ↓
START NAVIGATION
 ↓
REACH DESTINATION
 ↓
COMPLETE ACTIVITY
 ↓
EARN XP
 ↓
UNLOCK BADGE
 ↓
UPDATE LEADERBOARD
```

This creates a feedback loop that encourages users to actively explore the campus.

---

# 🚀 Future Scope

## Campus Data

* More buildings
* More campus spots
* Better location descriptions
* Regular campus data updates

## Navigation

* More accurate routing
* Accessibility-aware routes
* Real-time navigation
* Voice guidance

## Gamification

* More challenge types
* Achievement systems
* Advanced leaderboards
* Team-based challenges
* Event-based challenges
* Reward systems

## Platform

* Mobile application
* PostgreSQL for larger deployments
* University system integration
* QR-based location discovery

---

# 🔐 Scope

SNU Surfer intentionally focuses on:

```text
BUILDINGS
    +
CAMPUS SPOTS
    +
LOCATION INFORMATION
    +
INTERACTIVE MAP
    +
NAVIGATION
    +
GAMIFICATION
    +
DEVOPS
```

The application is designed around **campus-level destinations and geographic locations**.

---

# 👥 Team

**Team Members**

* `[ADD NAME]`
* `[ADD NAME]`
* `[ADD NAME]`
* `[ADD NAME]`

**Institution:** Shiv Nadar University Chennai
**Department:** Computer Science / Cyber Security
**Course:** `[ADD COURSE]`

---

# 📌 Project Status

**Status:** 🚧 In Development

Current development priorities:

* [ ] Define campus location schema
* [ ] Populate verified campus location data
* [ ] Implement Flask backend
* [ ] Implement REST APIs
* [ ] Build interactive Leaflet.js map
* [ ] Implement location search
* [ ] Implement navigation logic
* [ ] Design gamification system
* [ ] Implement XP / achievements
* [ ] Implement automated tests
* [ ] Create Docker environment
* [ ] Configure GitHub Actions CI/CD
* [ ] Deploy application

---

# 🏁 Conclusion

**SNU Surfer** aims to transform campus navigation from a basic location-finding task into an interactive campus exploration experience.

By combining a structured campus location database, searchable information, interactive mapping, navigation, gamification, and DevOps practices, the system provides a foundation for a smarter and more engaging campus experience.

```text
BUILDINGS + CAMPUS SPOTS
          ↓
       SEARCH
          ↓
         MAP
          ↓
      NAVIGATION
          ↓
     GAMIFICATION
          ↓
        DEVOPS
```

## SNU SURFER

**Find it. Navigate it. Explore it.**
