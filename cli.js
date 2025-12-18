#!/usr/bin/env node

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');
const os = require('os');

const SKILLS_DIR = path.join(__dirname, 'skills');
const CLAUDE_SKILLS_DIR = path.join(os.homedir(), '.claude', 'skills');

const colors = {
  reset: '\x1b[0m',
  bold: '\x1b[1m',
  dim: '\x1b[2m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  cyan: '\x1b[36m',
  red: '\x1b[31m'
};

function log(msg) { console.log(msg); }
function success(msg) { console.log(`${colors.green}${colors.bold}${msg}${colors.reset}`); }
function info(msg) { console.log(`${colors.cyan}${msg}${colors.reset}`); }
function warn(msg) { console.log(`${colors.yellow}${msg}${colors.reset}`); }
function error(msg) { console.log(`${colors.red}${msg}${colors.reset}`); }

function loadSkillsJson() {
  const skillsJsonPath = path.join(__dirname, 'skills.json');
  if (fs.existsSync(skillsJsonPath)) {
    return JSON.parse(fs.readFileSync(skillsJsonPath, 'utf8'));
  }
  return { skills: [] };
}

function getAvailableSkills() {
  if (!fs.existsSync(SKILLS_DIR)) return [];
  return fs.readdirSync(SKILLS_DIR).filter(name => {
    const skillPath = path.join(SKILLS_DIR, name);
    return fs.statSync(skillPath).isDirectory() &&
           fs.existsSync(path.join(skillPath, 'SKILL.md'));
  });
}

function installSkill(skillName) {
  const sourcePath = path.join(SKILLS_DIR, skillName);
  const destPath = path.join(CLAUDE_SKILLS_DIR, skillName);

  if (!fs.existsSync(sourcePath)) {
    error(`Skill "${skillName}" not found.`);
    log(`\nAvailable skills:`);
    listSkills();
    return false;
  }

  // Create .claude/skills directory if it doesn't exist
  if (!fs.existsSync(CLAUDE_SKILLS_DIR)) {
    fs.mkdirSync(CLAUDE_SKILLS_DIR, { recursive: true });
  }

  // Copy skill directory
  copyDir(sourcePath, destPath);

  success(`\nInstalled: ${skillName}`);
  info(`Location: ${destPath}`);
  log(`\nThe skill is now available in Claude Code, Cursor, and other compatible agents.`);
  return true;
}

function copyDir(src, dest) {
  if (fs.existsSync(dest)) {
    fs.rmSync(dest, { recursive: true });
  }
  fs.mkdirSync(dest, { recursive: true });

  const entries = fs.readdirSync(src, { withFileTypes: true });
  for (const entry of entries) {
    const srcPath = path.join(src, entry.name);
    const destPath = path.join(dest, entry.name);
    if (entry.isDirectory()) {
      copyDir(srcPath, destPath);
    } else {
      fs.copyFileSync(srcPath, destPath);
    }
  }
}

function listSkills() {
  const data = loadSkillsJson();
  const skills = data.skills || [];

  if (skills.length === 0) {
    warn('No skills found in skills.json');
    return;
  }

  // Group by category
  const byCategory = {};
  skills.forEach(skill => {
    const cat = skill.category || 'other';
    if (!byCategory[cat]) byCategory[cat] = [];
    byCategory[cat].push(skill);
  });

  log(`\n${colors.bold}Available Skills${colors.reset} (${skills.length} total)\n`);

  Object.keys(byCategory).sort().forEach(category => {
    log(`${colors.blue}${colors.bold}${category.toUpperCase()}${colors.reset}`);
    byCategory[category].forEach(skill => {
      const featured = skill.featured ? ` ${colors.yellow}*${colors.reset}` : '';
      log(`  ${colors.green}${skill.name}${colors.reset}${featured}`);
      log(`    ${colors.dim}${skill.description.slice(0, 70)}...${colors.reset}`);
    });
    log('');
  });

  log(`${colors.dim}* = featured skill${colors.reset}`);
  log(`\nInstall: ${colors.cyan}npx ai-agent-skills install <skill-name>${colors.reset}`);
}

function searchSkills(query) {
  const data = loadSkillsJson();
  const skills = data.skills || [];
  const q = query.toLowerCase();

  const matches = skills.filter(s =>
    s.name.toLowerCase().includes(q) ||
    s.description.toLowerCase().includes(q) ||
    (s.category && s.category.toLowerCase().includes(q))
  );

  if (matches.length === 0) {
    warn(`No skills found matching "${query}"`);
    return;
  }

  log(`\n${colors.bold}Search Results${colors.reset} (${matches.length} matches)\n`);
  matches.forEach(skill => {
    log(`${colors.green}${skill.name}${colors.reset} ${colors.dim}[${skill.category}]${colors.reset}`);
    log(`  ${skill.description.slice(0, 80)}...`);
    log('');
  });
}

function showHelp() {
  log(`
${colors.bold}AI Agent Skills${colors.reset}
The universal skill repository for AI agents.

${colors.bold}Usage:${colors.reset}
  npx ai-agent-skills <command> [options]

${colors.bold}Commands:${colors.reset}
  ${colors.green}list${colors.reset}              List all available skills
  ${colors.green}install <name>${colors.reset}   Install a skill to ~/.claude/skills/
  ${colors.green}search <query>${colors.reset}   Search skills by name, description, or category
  ${colors.green}info <name>${colors.reset}      Show detailed info about a skill
  ${colors.green}help${colors.reset}              Show this help message

${colors.bold}Examples:${colors.reset}
  npx ai-agent-skills list
  npx ai-agent-skills install pdf
  npx ai-agent-skills install frontend-design
  npx ai-agent-skills search excel

${colors.bold}Compatible with:${colors.reset}
  Claude Code, Cursor, Amp, VS Code, GitHub Copilot, Goose, Letta, OpenCode

${colors.bold}More info:${colors.reset}
  https://skillcreator.ai/discover
  https://github.com/skillcreatorai/Ai-Agent-Skills
`);
}

function showInfo(skillName) {
  const data = loadSkillsJson();
  const skill = data.skills.find(s => s.name === skillName);

  if (!skill) {
    error(`Skill "${skillName}" not found.`);
    return;
  }

  log(`
${colors.bold}${skill.name}${colors.reset}${skill.featured ? ` ${colors.yellow}(featured)${colors.reset}` : ''}

${colors.dim}${skill.description}${colors.reset}

${colors.bold}Category:${colors.reset}  ${skill.category}
${colors.bold}Author:${colors.reset}    ${skill.author}
${colors.bold}License:${colors.reset}   ${skill.license}
${colors.bold}Source:${colors.reset}    ${skill.source}
${colors.bold}Stars:${colors.reset}     ${skill.stars.toLocaleString()}
${colors.bold}Downloads:${colors.reset} ${skill.downloads.toLocaleString()}

${colors.bold}Install:${colors.reset}
  npx ai-agent-skills install ${skill.name}
`);
}

// Main CLI
const args = process.argv.slice(2);
const command = args[0] || 'help';
const param = args[1];

switch (command) {
  case 'list':
  case 'ls':
    listSkills();
    break;
  case 'install':
  case 'i':
    if (!param) {
      error('Please specify a skill name.');
      log('Usage: npx agent-skills install <skill-name>');
      process.exit(1);
    }
    installSkill(param);
    break;
  case 'search':
  case 's':
    if (!param) {
      error('Please specify a search query.');
      log('Usage: npx agent-skills search <query>');
      process.exit(1);
    }
    searchSkills(param);
    break;
  case 'info':
    if (!param) {
      error('Please specify a skill name.');
      log('Usage: npx agent-skills info <skill-name>');
      process.exit(1);
    }
    showInfo(param);
    break;
  case 'help':
  case '--help':
  case '-h':
    showHelp();
    break;
  default:
    // If command looks like a skill name, try to install it
    if (getAvailableSkills().includes(command)) {
      installSkill(command);
    } else {
      error(`Unknown command: ${command}`);
      showHelp();
      process.exit(1);
    }
}
